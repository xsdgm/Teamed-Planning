"""Frozen, offline abstract-state evaluation and conservative baseline fallback.

Calibration here is empirical across three named locations, not a safety or
coverage guarantee for a new environment. It never enables aerial deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));__package__='world_model'

import numpy as np
import torch
from torch.utils.data import DataLoader

from .abstract_data import AbstractTasks, load_prepared, utility, evidence_state, corridor
from .abstract_model import load_checkpoint
from .abstract_training import evaluate, baselines, baseline_scores, choice_metrics
from .semantic_data import atomic_json, sha256


def groups_for(scenes,arrays):
    return np.array([scenes[int(i)]['group'] for i in arrays['scene']])


def calibrate_fallback(scores,baseline,arrays,groups,*,quantile=.9,minimum_disagreements=20):
    """Bound overestimated advantage of the actually selected proposal.

    A worst-place bound avoids pooling large/easy places over smaller/harder
    ones. Test labels never enter this function. Three calibration places are
    still too few for a guarantee beyond this descriptive offline experiment.
    """
    if not np.isfinite(scores).all() or not np.isfinite(baseline).all():
        raise ValueError('Nonfinite calibration scores')
    proposed=scores.argmax(1);fallback=baseline.argmax(1);rows=np.arange(len(scores))
    predicted=scores[rows,proposed]-scores[rows,fallback]
    truth=utility(arrays['target'],arrays['state'],arrays['costs'])[:,:,0]
    actual=truth[rows,proposed]-truth[rows,fallback]
    errors=predicted-actual;records=[]
    for group in sorted(set(groups)):
        selected=(groups==group)&(proposed!=fallback)
        values=np.sort(errors[selected]);n=len(values)
        # Conservative finite-sample rank, without claiming exchangeability.
        rank=min(n-1,int(np.ceil((n+1)*quantile))-1) if n else 0
        records.append(dict(group=group,disagreements=n,
            bound=max(0.,float(values[rank])) if n else None,
            insufficient_support=n<minimum_disagreements))
    enabled=bool(records) and all(not x['insufficient_support'] for x in records)
    return dict(format='goose-abstract-fallback-v1',enabled=enabled,
        overestimation_bound=max(x['bound'] for x in records) if enabled else None,
        quantile=quantile,minimum_disagreements=minimum_disagreements,groups=records,
        action_domain='offline_observation_windows',deployment_ready=False,
        description='empirical worst-location bound, not an out-of-distribution or real navigation guarantee')


def guarded_choices(scores,baseline,calibration):
    """Unknown/unsupported advantage returns the frozen baseline selection."""
    scores=np.asarray(scores);baseline=np.asarray(baseline)
    if scores.shape!=baseline.shape or scores.ndim!=2 or not np.isfinite(baseline).all():
        raise ValueError('Invalid fallback baseline scores')
    fallback=baseline.argmax(1);finite=np.isfinite(scores).all(1)
    safe=np.where(finite[:,None],scores,0.);proposed=safe.argmax(1)
    rows=np.arange(len(scores));margin=safe[rows,proposed]-safe[rows,fallback]
    bound=calibration.get('overestimation_bound')
    enabled=(calibration.get('format')=='goose-abstract-fallback-v1' and calibration.get('enabled') is True
        and calibration.get('action_domain')=='offline_observation_windows' and calibration.get('deployment_ready') is False
        and isinstance(bound,(float,int)) and np.isfinite(bound) and bound>=0)
    override=(finite&(proposed!=fallback)&(margin>bound+1e-6)) if enabled else np.zeros(len(scores),bool)
    return np.where(override,proposed,fallback),override


def guard_metrics(scores,baseline,arrays,calibration):
    selected,override=guarded_choices(scores,baseline,calibration)
    rows=np.arange(len(selected));fallback=baseline.argmax(1)
    truth=utility(arrays['target'],arrays['state'],arrays['costs'])[:,:,0]
    gain=truth[rows,selected]-truth[rows,fallback]
    selection=np.full(scores.shape,-1.,np.float32);selection[rows,selected]=1.
    return dict(ranking=choice_metrics(selection,arrays),overrides=int(override.sum()),
        override_coverage=float(override.mean()),fallback_fraction=float((~override).mean()),
        mean_utility_gain_over_baseline=float(gain.mean()),
        harmful_overrides=int((override&(gain< -1e-6)).sum()),
        harmful_override_fraction=float((gain[override]< -1e-6).mean()) if override.any() else None,
        all_abstain_is_not_improvement=bool(not override.any()))


def connectivity_metrics(prediction,scores,arrays):
    rows=np.arange(len(scores));chosen=scores.argmax(1)
    opportunity=(arrays['target'][:,:,0,0]>arrays['state'][:,None,0]).any(1)
    connected=arrays['target'][rows,chosen,0,0]>arrays['state'][:,0]
    per_step=[]
    for step in range(2):
        truth=arrays['target'][:,:,step,0];p=prediction[:,:,step,0];positive=truth>.5
        per_step.append(dict(positive_targets=int(positive.sum()),negative_targets=int((~positive).sum()),
            positive_brier=float(((p-truth)**2)[positive].mean()) if positive.any() else None,
            negative_brier=float(((p-truth)**2)[~positive].mean()) if (~positive).any() else None,
            mean_prediction_on_positive=float(p[positive].mean()) if positive.any() else None,
            mean_prediction_on_negative=float(p[~positive].mean()) if (~positive).any() else None))
    return dict(tasks_with_connecting_candidate=int(opportunity.sum()),selected_connecting_candidate=int(connected.sum()),
        opportunity_selection_rate=float(connected.sum()/opportunity.sum()) if opportunity.any() else None,per_step=per_step)


@torch.no_grad()
def wrong_goal_scores(model,arrays,device):
    """Diagnostic ablation: feed a reflected goal, evaluate the original task."""
    maps=arrays['maps'].copy();task=arrays['task'].copy();state=[]
    for i in range(len(maps)):
        start=arrays['start'][i];goal=15-arrays['goal'][i]
        task[i,2:]=goal/15*2-1;maps[i,8]=0;maps[i,8,goal[1],goal[0]]=1
        maps[i,9]=corridor(start,goal)
        state.append(evidence_state(maps[i,:5],maps[i,5]>.5,start,goal))
    state=np.array(state);predictions=[]
    for begin in range(0,len(maps),32):
        batch=slice(begin,begin+32)
        values=[torch.tensor(value[batch],device=device) for value in (maps,state,task,arrays['actions'])]
        predictions.append(model.predict(*values)['state_mean'].cpu().numpy())
    return utility(np.concatenate(predictions),state,arrays['costs'])[:,:,0]


def assess(run_root,*,device='cpu'):
    root=Path(run_root).resolve();output=root/'assessment'
    if output.exists():raise FileExistsError('Evaluation is write-once; inspect existing assessment first')
    protocol=json.loads((root/'protocol.json').read_text())
    runs=[root/f'seed-{seed}' for seed in protocol['seeds']]
    for run in runs:
        if json.loads((run/'status.json').read_text())['stage']!='complete':raise ValueError('Training incomplete')
    configs=[json.loads((run/'config.json').read_text()) for run in runs]
    checkpoints=[]
    for run in runs:
        _,doc=load_checkpoint(run/'best.pt');checkpoints.append(doc)
    # This selection is frozen before loading calibration/test task arrays.
    index=min(range(len(runs)),key=lambda i:checkpoints[i]['validation']['selection_score'])
    selected=runs[index];config=configs[index]
    if len({x['prepared_manifest_sha256'] for x in configs})!=1:raise ValueError('Mismatched experiment data')
    if any(x['best_validation_baseline']!=config['best_validation_baseline'] for x in configs):raise ValueError('Baseline mismatch')
    output.mkdir()
    atomic_json(output/'selection.json',dict(run=str(selected),checkpoint_sha256=sha256(selected/'best.pt'),
        baseline=config['best_validation_baseline'],criterion='validation scaled state MSE + one-step proxy regret',
        protocol_sha256=sha256(root/'protocol.json')))
    shutil.copy2(Path(__file__),output/'abstract_assessment.py')
    torch.set_num_threads(2)
    scenes,arrays,data=load_prepared(config['prepared_data'],config['source_manifest'])
    if sha256(Path(config['prepared_data'])/'manifest.json')!=config['prepared_manifest_sha256']:
        raise ValueError('Prepared data changed since training')
    with np.load(selected/'ridge.npz') as z:coefficients=z['coefficients']
    scales=np.array(config['scales'],np.float32);baseline_name=config['best_validation_baseline']
    model,_=load_checkpoint(selected/'best.pt',device)
    predictions={};metrics={}
    for split in ('calibration','test'):
        loader=DataLoader(AbstractTasks(scenes[split],arrays[split],include_future=False),batch_size=32)
        result,members=evaluate(model,loader,arrays[split],scales,device,return_members=True)
        predictions[split]=members.mean(0)
        np.savez_compressed(output/(split+'_predictions.npz'),state_members=members)
        metrics[split]=dict(model=result,baselines=baselines(arrays[split],coefficients,scales))
    scores={split:utility(p,arrays[split]['state'],arrays[split]['costs'])[:,:,0] for split,p in predictions.items()}
    baseline={split:baseline_scores(baseline_name,arrays[split],coefficients) for split in scores}
    calibration=calibrate_fallback(scores['calibration'],baseline['calibration'],arrays['calibration'],groups_for(scenes['calibration'],arrays['calibration']))
    atomic_json(output/'calibration.json',calibration)
    for split in scores:
        metrics[split]['connectivity']=connectivity_metrics(predictions[split],scores[split],arrays[split])
        metrics[split]['guard']=guard_metrics(scores[split],baseline[split],arrays[split],calibration)
        groups=groups_for(scenes[split],arrays[split]);by_group={}
        for group in sorted(set(groups)):
            mask=np.flatnonzero(groups==group);a={key:value[mask] for key,value in arrays[split].items()}
            by_group[group]=dict(model=choice_metrics(scores[split],arrays[split],mask),
                baseline=choice_metrics(baseline[split],arrays[split],mask),
                guard=guard_metrics(scores[split][mask],baseline[split][mask],a,calibration))
        metrics[split]['by_group']=by_group
    metrics['test']['diagnostics']=dict(
        wrong_goal=choice_metrics(wrong_goal_scores(model,arrays['test'],device),arrays['test']),
        broken_action_correspondence=choice_metrics(np.roll(scores['test'],1,axis=1),arrays['test']))
    seed_results=[]
    for i,run in enumerate(runs):
        if i==index:result=metrics['test']['model']
        else:
            other,_=load_checkpoint(run/'best.pt',device)
            loader=DataLoader(AbstractTasks(scenes['test'],arrays['test'],include_future=False),batch_size=32)
            result,members=evaluate(other,loader,arrays['test'],scales,device,return_members=True)
            np.savez_compressed(output/(run.name+'_test_predictions.npz'),state_members=members)
        seed_results.append(dict(seed=configs[i]['seed'],best_epoch=checkpoints[i]['epoch'],
            validation=checkpoints[i]['validation'],test=result,checkpoint_sha256=sha256(run/'best.pt')))
    validation_baselines=json.loads((selected/'validation_baselines.json').read_text())
    state_baseline=min((name for name in validation_baselines if 'scaled_mse' in validation_baselines[name]),
        key=lambda name:validation_baselines[name]['scaled_mse'])
    test=metrics['test'];model_utility=test['model']['ranking']['mean_proxy_utility']
    baseline_utility=test['baselines'][baseline_name]['ranking']['mean_proxy_utility']
    report=dict(format='goose-abstract-assessment-v1',selected_run=str(selected),selected_seed=config['seed'],
        ranking_baseline=baseline_name,state_baseline=state_baseline,metrics=metrics,seeds=seed_results,
        conclusions=dict(state_mse_better_than_validation_selected_baseline=test['model']['scaled_mse']<test['baselines'][state_baseline]['scaled_mse'],
            raw_model_mean_utility_gain=model_utility-baseline_utility,
            raw_model_better_on_all_test_places=all(x['model']['mean_proxy_utility']>x['baseline']['mean_proxy_utility'] for x in test['by_group'].values()),
            fallback_mean_utility_gain=test['guard']['mean_utility_gain_over_baseline'],
            deployment_ready=False,world_model_advice_enabled=False),
        limitations=['No real drone action-to-returned-map trajectories or vehicle clearance ground truth.',
            'Named-place splits are not GPS-verified spatial isolation; GOOSE-Ex geography remains unresolved and is training-only.',
            'Virtual start/goal and masked observations of static ground scans; no physical flight cost.',
            'Tasks require a measured open-ground semantic anchor; scans without such an anchor are excluded, limiting dense-forest applicability.',
            'Few independent places and rare positive connectivity events; sample averages are descriptive.',
            'Three-member disagreement is not calibrated semantic certainty; abstention concerns action advantage.',
            'Public GOOSE scenes reused for a new task; no untouched blind navigation claim.',
            'Frozen vtmp/v9 and runtime defaults remain unchanged; no WildScenes migration.'])
    atomic_json(output/'report.json',report)
    atomic_json(output/'status.json',dict(stage='complete',report_sha256=sha256(output/'report.json'),world_model_advice_enabled=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args()
    report=assess(**vars(args));print(json.dumps(report['conclusions'],indent=2))


if __name__=='__main__':main()
