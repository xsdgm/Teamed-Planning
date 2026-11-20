"""Train and select abstract latent observation transitions on GOOSE only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));__package__='world_model'

import numpy as np
import torch
from torch.utils.data import DataLoader

from .abstract_data import (AbstractTasks, STATE_NAMES, TRAVEL_WEIGHT,
    baseline_features, load_prepared, prepare, utility, query_mask)
from .abstract_model import AbstractWorldModel, ARCHITECTURE, FORMAT, SCHEMA_VERSION, loss
from .hierarchical_sources import atomic_checkpoint, copy_source_files
from .semantic_data import atomic_json, sha256

V9_SHA256='7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198'
SOURCES=('world_model/__init__.py', 'world_model/hierarchical_sources.py', 'world_model/terrain/abstract_data.py','world_model/terrain/abstract_model.py','world_model/abstract_training.py',
         'world_model/goose_data.py','world_model/terrain/frontier_data.py','world_model/semantic_data.py')


def checkpoint_save(path, document):
    return atomic_checkpoint(path, document)


def choice_metrics(scores,arrays,indices=None):
    if indices is None:indices=np.arange(len(scores))
    a={k:v[indices] for k,v in arrays.items()};scores=scores[indices]
    truth=utility(a['target'],a['state'],a['costs'])[:,:,0]
    choice=scores.argmax(1);rows=np.arange(len(choice));chosen=truth[rows,choice]
    oracle=truth.max(1);spread=np.ptp(truth,axis=1);informative=spread>.05
    return dict(tasks=len(choice),mean_proxy_utility=float(chosen.mean()),mean_regret=float((oracle-chosen).mean()),
        oracle_mean=float(oracle.mean()),informative_tasks=int(informative.sum()),
        informative_regret=float((oracle-chosen)[informative].mean()) if informative.any() else None,
        informative_best_choice_rate=float((oracle-chosen<=1e-6)[informative].mean()) if informative.any() else None,
        chosen_new_connections=int((a['target'][rows,choice,0,0]>a['state'][:,0]).sum()))


def state_metrics(prediction,arrays,scales):
    error=(prediction-arrays['target'])**2
    step_error=error.mean((0,1))
    score=utility(prediction,arrays['state'],arrays['costs'])[:,:,0]
    choice=choice_metrics(score,arrays)
    return dict(scaled_mse=float((error/scales**2).mean()),
        per_step_state_mse=[dict(zip(STATE_NAMES,row.tolist())) for row in step_error],
        connected_brier=float(error[...,0].mean()),connection_positive_targets=int(arrays['target'][...,0].sum()),
        ranking=choice,selection_score=float((error/scales**2).mean()+choice['mean_regret']))


@torch.no_grad()
def evaluate(model,loader,arrays,scales,device,*,return_members=False):
    model.eval();predictions=[]
    for batch in loader:
        result=model.predict(*(batch[key].to(device) for key in ('maps','state','task','actions')))
        predictions.append(result['state_members'].cpu().numpy())
    members=np.concatenate(predictions,axis=1)
    metrics=state_metrics(members.mean(0),arrays,scales)
    return (metrics,members) if return_members else metrics


def fit_ridge(training,validation,scales):
    coefficients=[];regularizers=[]
    for step in range(2):
        x=baseline_features(training,step).reshape(-1,23)
        vx=baseline_features(validation,step)
        y=(training['target'][:,:,step]-training['state'][:,None]).reshape(-1,len(STATE_NAMES))
        best=None
        for reg in (.001,.1,1.,10.,100.):
            penalty=np.eye(x.shape[1])*reg;penalty[0,0]=0
            coef=np.linalg.solve(x.T@x+penalty,x.T@y)
            p=np.clip(validation['state'][:,None]+vx@coef,validation['state'][:,None],1.)
            error=float((((p-validation['target'][:,:,step])/scales)**2).mean())
            if best is None or error<best[0]:best=(error,reg,coef)
        coefficients.append(best[2]);regularizers.append(best[1])
    return np.stack(coefficients),regularizers


def ridge_predict(arrays,coefficients):
    outputs=[]
    for step in range(2):
        p=np.clip(arrays['state'][:,None]+baseline_features(arrays,step)@coefficients[step],arrays['state'][:,None],1.)
        p[...,2]=np.minimum(p[...,2],p[...,0]);p[...,1]=np.maximum(p[...,1],p[...,0])
        p[...,4]=np.minimum(p[...,4],p[...,3]);p[...,6]=np.maximum(p[...,6],p[...,0])
        outputs.append(p)
    return np.stack(outputs,2).astype(np.float32)


def baselines(arrays,coefficients,scales):
    persistence=np.broadcast_to(arrays['state'][:,None,None],arrays['target'].shape).copy()
    results={name:state_metrics(p,arrays,scales) for name,p in
        (('persistence',persistence),('optimistic_goal_graph',arrays['optimistic']),('ridge',ridge_predict(arrays,coefficients)))}
    coverage=np.array([[((query_mask(action)&~known).sum()) for action in branch[:,0]]
                       for known,branch in zip(arrays['known'],arrays['actions_grid'])],np.float32)
    coverage=coverage/256-TRAVEL_WEIGHT*arrays['costs'][:,:,0]
    results['generic_coverage']=dict(ranking=choice_metrics(coverage,arrays))
    return results


def baseline_scores(name,arrays,coefficients):
    if name=='generic_coverage':
        coverage=np.array([[((query_mask(action)&~known).sum()) for action in branch[:,0]]
            for known,branch in zip(arrays['known'],arrays['actions_grid'])],np.float32)
        return coverage/256-TRAVEL_WEIGHT*arrays['costs'][:,:,0]
    prediction=(ridge_predict(arrays,coefficients) if name=='ridge' else arrays['optimistic'] if name=='optimistic_goal_graph'
                else np.broadcast_to(arrays['state'][:,None,None],arrays['target'].shape))
    return utility(prediction,arrays['state'],arrays['costs'])[:,:,0]


def train(data_manifest,output,*,v9_checkpoint,prepared_data=None,epochs=60,patience=12,batch_size=16,
          hidden=24,latent=96,members=3,device='cuda',seed=20261114):
    output=Path(output).resolve()
    if (output/'last.pt').exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    if sha256(v9_checkpoint)!=V9_SHA256:raise ValueError('Frozen v9 identity mismatch')
    if min(epochs,batch_size,patience)<1:raise ValueError('Positive training counts required')
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    atomic_json(output/'status.json',dict(stage='loading_offline_tasks',pid=os.getpid()))
    if prepared_data:
        scenes,arrays,data=load_prepared(prepared_data,data_manifest)
    else:
        prepared_data=output/'data';scenes,arrays,data=prepare(data_manifest,prepared_data)
    trainset=AbstractTasks(scenes['train'],arrays['train'],augment=True,seed=seed)
    valset=AbstractTasks(scenes['val'],arrays['val'],include_future=False)
    training=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
    validation=DataLoader(valset,batch_size=batch_size,shuffle=False,num_workers=0)
    differences=arrays['train']['target']-arrays['train']['state'][:,None,None]
    scales=np.maximum(differences.std((0,1,2)),.05).astype(np.float32)
    coefficients,regularizers=fit_ridge(arrays['train'],arrays['val'],scales)
    np.savez(output/'ridge.npz',coefficients=coefficients,regularizers=regularizers)
    baseline=baselines(arrays['val'],coefficients,scales)
    atomic_json(output/'validation_baselines.json',baseline)
    best_baseline=max(baseline,key=lambda name:baseline[name]['ranking']['mean_proxy_utility'])
    model=AbstractWorldModel(hidden,latent,members).to(device)
    config=dict(format=FORMAT,schema_version=SCHEMA_VERSION,model_architecture=ARCHITECTURE,
        architecture=model.configuration(),states=list(STATE_NAMES),epochs=epochs,patience=patience,
        seed=seed,batch_size=batch_size,device=device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        prepared_data=str(Path(prepared_data).resolve()),prepared_manifest_sha256=sha256(Path(prepared_data)/'manifest.json'),
        source_manifest=str(Path(data_manifest).resolve()),source_manifest_sha256=sha256(data_manifest),
        scales=scales.tolist(),best_validation_baseline=best_baseline,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        reconstruction_loss=False,latent_transition_loss=True,action_horizon=2,
        task_objective='goal-conditioned semantic evidence proxy changes after offline observation windows',
        simulation_used=False,offline_mask_augmentation=True,real_aerial_actions=False,
        graph_is_vehicle_traversability=False,world_model_advice_enabled=False,deployment_ready=False,
        v9_weights_trained=False,v9_checkpoint_sha256=V9_SHA256)
    atomic_json(output/'config.json',config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=3e-5)
    scales_tensor=torch.tensor(scales,device=device)
    best=float('inf');best_epoch=0;updates=0;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();trainset.epoch=epoch;total=0.;components=np.zeros(4)
        for step,batch in enumerate(training,1):
            batch={key:value.to(device) for key,value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            objective,parts=loss(model,batch,scales_tensor)
            if not torch.isfinite(objective):raise ValueError('Nonfinite abstract loss')
            objective.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();model.update_teacher();updates+=1
            total+=float(objective.detach());components+=parts.detach().cpu().numpy()
            if step%16==0 or step==1:
                atomic_json(output/'status.json',dict(stage='training',pid=os.getpid(),epoch=epoch,epochs=epochs,
                    batch=step,batches=len(training),optimizer_steps=updates,best_epoch=best_epoch,
                    elapsed_seconds=time.monotonic()-started))
        schedule.step();metrics=evaluate(model,validation,arrays['val'],scales,device)
        improved=metrics['selection_score']<best-1e-5
        if improved:best=metrics['selection_score'];best_epoch=epoch
        row=dict(epoch=epoch,optimizer_steps=updates,train_loss=total/len(training),
            loss_components=dict(zip(('scaled_state','connected_bce','latent_consistency','ranking_kl'),(components/len(training)).tolist())),
            validation=metrics,best_epoch=best_epoch,best_score=best,gradient_norm=float(gradient),elapsed_seconds=time.monotonic()-started)
        with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        document={**config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
                  'epoch':epoch,'optimizer_steps':updates,'validation':metrics}
        checkpoint_save(output/'last.pt',document)
        if improved:checkpoint_save(output/'best.pt',document)
        atomic_json(output/'status.json',dict(stage='training',pid=os.getpid(),**row))
        print(json.dumps(dict(epoch=epoch,loss=row['train_loss'],validation_score=metrics['selection_score'],
            regret=metrics['ranking']['mean_regret'],best_epoch=best_epoch)),flush=True)
        if epoch-best_epoch>=patience:break
    result=dict(stage='complete',epochs=epoch,epochs_requested=epochs,best_epoch=best_epoch,best_score=best,
        optimizer_steps=updates,stop_reason='validation_no_improvement' if epoch<epochs else 'epoch_budget_completed',
        elapsed_seconds=time.monotonic()-started,real_aerial_actions=False,deployment_ready=False,
        world_model_advice_enabled=False,v9_unchanged=sha256(v9_checkpoint)==V9_SHA256)
    atomic_json(output/'status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--v9-checkpoint',type=Path,required=True);parser.add_argument('--prepared-data',type=Path)
    for name,default in (('epochs',60),('patience',12),('batch_size',16),('hidden',24),('latent',96),('members',3),('seed',20261114)):
        parser.add_argument('--'+name.replace('_','-'),type=int,default=default)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    hashes = copy_source_files(source, output / 'sources', SOURCES)
    atomic_json(output/'source_manifest.json',hashes)
    atomic_json(output/'launch.json',dict(pid=os.getpid(),command=sys.argv))
    try:train(**vars(args))
    except BaseException as exc:
        atomic_json(output/'status.json',dict(stage='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__':main()
