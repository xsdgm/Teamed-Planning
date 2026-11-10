"""Detached vtmp2 scene certification, data collection and GPU training."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

# Works with python -I and from the immutable per-run source snapshot.
SOURCE = Path(__file__).resolve().parent
if __package__ in (None, ''):
    sys.path.insert(0, str(SOURCE.parent))
    __package__ = 'world_model'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(job):
    from .feasibility import generate_certified_scene
    return generate_certified_scene(**job)


def collect(job):
    from .data import collect_episode
    return collect_episode(job)


def bootstrap_ready(completed, train_count=4, val_count=2):
    return (sum(item['split']=='train' for item in completed) >= train_count
            and sum(item['split']=='val' for item in completed) >= val_count)


def freeze_bootstrap(completed, destination, train_count=4, val_count=2):
    """Freeze an explicit subset; future collection cannot change its dataset."""
    destination = Path(destination)
    destination.mkdir()
    selected = []
    for split, count in (('train', train_count), ('val', val_count)):
        eligible = sorted((x for x in completed if x['split']==split), key=lambda x:x['seed'])
        if len(eligible)<count:
            raise ValueError('Bootstrap requires complete training and validation episodes')
        for item in eligible[:count]:
            source = Path(item['file'])
            os.link(source, destination/source.name)
            summary = source.with_suffix('.json')
            if summary.exists():
                os.link(summary, destination/summary.name)
            selected.append({'seed': item['seed'], 'split': split, 'sha256': digest(source)})
    write_json(destination/'selection.json', selected)


def run(config):
    output = Path(config['output'])
    started = time.time()
    state = dict(algorithm='vtmp2', pid=os.getpid(), started_at=started,
                 gpu_uuid=config['gpu_uuid'], train_scenes=config['train_scenes'],
                 val_scenes=config['val_scenes'], steps_per_scene=config['steps'],
                 epochs=config['epochs'], bootstrap_complete=False)

    def status(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        write_json(output/'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    with (output/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('preflight')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('A working CUDA installation is required for this run')
            x=torch.ones((32,32), device='cuda')
            if float((x@x)[0,0])!=32.:
                raise RuntimeError('CUDA matrix multiplication preflight failed')
            del x
            torch.cuda.empty_cache()
            from .data import provenance
            evidence = provenance()
            evidence.update(source_files={p.name:digest(p) for p in sorted(SOURCE.glob('*.py'))},
                            python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(0))
            write_json(output/'provenance.json', evidence)

            jobs = [dict(seed=config['train_seed_start']+i, split='train', output_dir=str(output), clearance=1.5)
                    for i in range(config['train_scenes'])]
            jobs += [dict(seed=config['val_seed_start']+i, split='val', output_dir=str(output), clearance=1.5)
                     for i in range(config['val_scenes'])]
            if len({job['seed'] for job in jobs})!=len(jobs):
                raise ValueError('Training and validation scene seeds must be disjoint')
            records = []
            status('screening', certified=0, total_scenes=len(jobs))
            context = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(config['workers'], mp_context=context) as pool:
                futures = {pool.submit(screen, job):job for job in jobs}
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    status('screening', certified=len(records), last_certified_seed=record['seed'])
            records.sort(key=lambda item:(item['split'], item['seed']))
            write_json(output/'scene_manifest.json', dict(
                algorithm='vtmp2', clearance_m=1.5, reference_path_in_policy=False,
                all_scenes_certified_before_collection=True, scenes=records))

            # Prioritize enough distinct scenes for a GPU optimization preflight.
            train_records=[x for x in records if x['split']=='train']
            val_records=[x for x in records if x['split']=='val']
            ordered=train_records[:4]+val_records[:2]+train_records[4:]+val_records[2:]
            dataset=output/'data'
            dataset.mkdir()
            completed=[]
            status('collecting', collected=0, total_scenes=len(ordered))
            with concurrent.futures.ProcessPoolExecutor(config['workers'], mp_context=context) as pool:
                futures={}
                for record in ordered:
                    job=dict(seed=record['seed'], split=record['split'], steps=config['steps'],
                             output=str(dataset), scene_path=record['scene_path'],
                             scene_sha256=record['scene_sha256'])
                    futures[pool.submit(collect,job)]=job
                for future in concurrent.futures.as_completed(futures):
                    item=future.result()
                    completed.append(item)
                    write_json(output/'collection_manifest.json', completed)
                    status('collecting', collected=len(completed), last_collected_seed=item['seed'])
                    if (config['bootstrap_epochs'] and not state['bootstrap_complete']
                            and bootstrap_ready(completed)):
                        bootstrap=output/'bootstrap_data'
                        freeze_bootstrap(completed, bootstrap)
                        status('bootstrap_training', collected=len(completed))
                        from .train import run_training
                        result=run_training(bootstrap, output/'bootstrap_training', device='cuda',
                            epochs=config['bootstrap_epochs'], batch_size=config['batch_size'],
                            sequence_length=config['sequence_length'], lr=config['lr'],
                            seed=config['training_seed'], bootstrap=True)
                        write_json(output/'bootstrap_result.json', result)
                        status('collecting', bootstrap_complete=True,
                               bootstrap_checkpoint=str(output/'bootstrap_training/latest.pt'))

            status('training', collected=len(completed), training_output=str(output/'training'))
            from .train import run_training
            result=run_training(dataset, output/'training', device='cuda', epochs=config['epochs'],
                batch_size=config['batch_size'], sequence_length=config['sequence_length'],
                lr=config['lr'], seed=config['training_seed'], bootstrap=False)
            write_json(output/'result.json', result)
            status('complete', result=result, wall_seconds=time.time()-started)
        except BaseException as exc:
            traceback.print_exc()
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def launch(args):
    output=Path(args.output).resolve()
    repository=SOURCE.parents[3]
    if output.parent != repository/'artifacts':
        raise ValueError('Run output must be directly under Teamed-Planning/artifacts for frozen-source paths')
    if output.exists():
        raise FileExistsError(f'Use a fresh run directory: {output}')
    output.mkdir(parents=True)
    snapshot=output/'runtime_sources/world_model'
    snapshot.mkdir(parents=True)
    for source in SOURCE.glob('*.py'):
        shutil.copy2(source, snapshot/source.name)
    shutil.copy2(SOURCE/'requirements-gpu.txt', snapshot/'requirements-gpu.txt')
    config=dict(output=str(output), gpu_uuid=args.gpu_uuid, workers=args.workers,
        train_scenes=args.train_scenes, val_scenes=args.val_scenes, steps=args.steps,
        epochs=args.epochs, bootstrap_epochs=args.bootstrap_epochs, batch_size=args.batch_size,
        sequence_length=args.sequence_length, lr=args.lr, train_seed_start=args.train_seed_start,
        val_seed_start=args.val_seed_start, training_seed=args.training_seed)
    write_json(output/'config.json', config)
    environment=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):
        environment.pop(name,None)
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (output/'pipeline.log').open('a', buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-I','-B','-u',str(snapshot/'pipeline.py'),
            '--run-config',str(output/'config.json')], cwd=output,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,env=environment)
    write_json(output/'launch.json',dict(pid=process.pid, launched_at=time.time(),
        python=sys.executable, config=str(output/'config.json'), log=str(output/'pipeline.log')))
    print(json.dumps(dict(pid=process.pid,output=str(output),log=str(output/'pipeline.log'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--train-scenes', type=int, default=32)
    parser.add_argument('--val-scenes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bootstrap-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train-seed-start', type=int, default=82000)
    parser.add_argument('--val-seed-start', type=int, default=83000)
    parser.add_argument('--training-seed', type=int, default=20261107)
    args=parser.parse_args()
    if args.run_config:
        run(json.loads(args.run_config.read_text()))
    else:
        if args.output is None or not args.gpu_uuid:
            parser.error('--output and --gpu-uuid are required for a new detached run')
        if (args.workers<1 or args.train_scenes<4 or args.val_scenes<2 or args.steps<10
                or args.steps%10 or args.epochs<1 or args.bootstrap_epochs<0
                or args.batch_size<1 or args.sequence_length<1 or not 0<args.lr<1):
            parser.error('Invalid run size, budget, sequence length or learning rate')
        launch(args)


if __name__=='__main__':
    main()
