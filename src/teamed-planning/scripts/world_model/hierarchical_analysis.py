"""Pilot ablation: does received evidence help beyond learned geometry alone?"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from world_model.hierarchical_sources import file_sha256
from world_model.navigation_evidence import ReturnModel,select


def evaluate(rows,model,columns):
    decisions=[]
    for row in rows:
        options=row['candidates']
        predicted=model.predict(np.array([q['features'] for q in options])[:,columns])
        choice=select(predicted,[q['status'] for q in options])
        actual=np.array([q['mean_return'] for q in options])
        if choice is None:
            decisions.append(dict(case=f"{row['case']['family']}_{row['case']['variant']}",selected=None))
        else:
            decisions.append(dict(case=f"{row['case']['family']}_{row['case']['variant']}",selected=choice,
                                  selected_return=float(actual[choice]),baseline_return=float(actual[0]),
                                  oracle_return=float(actual.max()),paired_gain=float(actual[choice]-actual[0])))
    valid=[d for d in decisions if d['selected'] is not None]
    return dict(decisions=decisions,mean_return=float(np.mean([d['selected_return'] for d in valid])) if valid else None)


def bootstrap_mean(values,seed=15112026):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(values,(5000,len(values)),replace=True).mean(1)
    return dict(mean=float(values.mean()),interval95=np.quantile(draws,[.025,.975]).tolist(),
                unit='scene context',contexts=len(values),method='paired context bootstrap; exploratory small sample')


def analyze(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    report=json.loads((output/'report.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json'))]
    split={s:[r for r in rows if r['case']['split']==s] for s in ('train','validation','test')}
    fits={}
    for name,columns in [('geometry_only',list(range(5))),('received_evidence',list(range(len(manifest['features']))))]:
        x=np.array([q['features'] for r in split['train'] for q in r['candidates']])[:,columns]
        y=np.array([q['mean_return'] for r in split['train'] for q in r['candidates']])
        options=[]
        for alpha in (.1,1.,10.,100.):
            model=ReturnModel(alpha).fit(x,y)
            options.append((evaluate(split['validation'],model,columns)['mean_return'],alpha,model))
        _,alpha,model=max(options,key=lambda v:v[0])
        result=evaluate(split['test'],model,columns)
        gains=[d['paired_gain'] for d in result['decisions'] if d['selected'] is not None]
        fits[name]=dict(regularization=alpha,test=result,paired_gain=bootstrap_mean(gains),
                        feature_names=[manifest['features'][i] for i in columns])
    # The evidence model must reproduce the independently recorded evaluator.
    expected=[d['selected'] for d in report['splits']['test']['decisions']]
    actual=[d['selected'] for d in fits['received_evidence']['test']['decisions']]
    if expected!=actual:raise AssertionError('Independent analysis disagrees with recorded selector')
    all_runs=[r for row in rows for q in row['candidates'] for r in q['rollouts']]
    metrics=dict(protocol=manifest['protocol'],ablation=fits,
                 source_sha256=file_sha256(__file__),
                 report_sha256=file_sha256(output/'report.json'),
                 total_executed_steps=sum(r['steps'] for r in all_runs),
                 mission_arrivals=sum(r['termination']=='mission_arrival' for r in all_runs),
                 horizons=sum(r['termination']=='horizon' for r in all_runs),
                 cbf_terminations=sum(r['termination']=='cbf_infeasible' for r in all_runs),
                 collisions=sum(r['collisions'] for r in all_runs),
                 subgoal_failure_events=sum(r['subgoal_failures'] for r in all_runs),
                 candidate_status_counts=dict(Counter(q['status'] for row in rows for q in row['candidates'])),
                 retrospective_best_gain_percent=100.*(report['splits']['test']['mean_oracle_return']/report['splits']['test']['mean_geometric_return']-1.),
                 limitation='This experiment estimates an 80-step return only when horizon=80. No full-mission or kilometer claim. Both learned models share the evidence feasibility mask; only their value features differ.')
    (output/'analysis.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze(args.output),ensure_ascii=False,indent=2))
