"""Generate a validated, single-seed E01/E03 comparison from genuine checkpoints."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from nano_lm.artifacts import atomic_json
from nano_lm.checkpoint import read_checkpoint
from nano_lm.results import collect


def compare(baseline, rope, root, out):
    out.mkdir(parents=True,exist_ok=False)
    runs=[]; payloads=[]
    for artifact,eid in [(baseline,'E01'),(rope,'E03')]:
        completion=json.loads((artifact/'completion.json').read_text())
        checkpoint=artifact/completion['final_checkpoint'];runs.append(checkpoint.parent.parent)
        payload,_=read_checkpoint(checkpoint)
        assert payload['metadata']['resolved']['experiment_id']==eid and payload['step']==5000
        payloads.append(payload)
    verified,aggregate=collect(runs,root,5000,require_complete=False)
    a,b=[p['metadata'] for p in payloads]
    assert a['resolved']['training']==b['resolved']['training']
    assert a['resolved']['tokenizer_hash']==b['resolved']['tokenizer_hash']
    assert a['resolved']['tokenizer_kind']==b['resolved']['tokenizer_kind']=='character'
    assert a['parameters']==b['parameters']==1804800
    ma,mb=a['resolved']['model'],b['resolved']['model']
    assert {k for k in ma if ma[k]!=mb[k]}=={'position'} and ma['position']=='sinusoidal' and mb['position']=='rope'
    freeze=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
    assert freeze(a['environment'])==freeze(b['environment'])
    assert all(a['environment'][k]==b['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
    histories=[p['training_state']['history'] for p in payloads]
    values=[[r for r in h if 'validation' in r] for h in histories]
    assert all([r['step'] for r in v]==list(range(250,5001,250)) for v in values)
    rows=[]
    for av,bv in zip(*values):
        row={'step':av['step']}
        for metric in ['loss','bpb','perplexity']:
            x,y=av['validation'][metric],bv['validation'][metric]
            row.update({f'E01_{metric}':x,f'E03_{metric}':y,f'delta_{metric}':y-x,f'relative_{metric}_percent':(y-x)/x*100})
        rows.append(row)
    summaries=[json.loads((p/'analysis/summary.json').read_text()) for p in [baseline,rope]]
    comparisons=[]
    for name,key,sub in [('best_loss','best_validation','loss'),('final_loss','final_validation','loss'),('best_bpb','best_validation','bpb'),('final_bpb','final_validation','bpb'),('final_training_loss','final_training_loss',None),('training_seconds','training_seconds',None),('validation_seconds','validation_seconds',None),('peak_gpu_bytes','peak_gpu_bytes',None),('parameters','parameters',None)]:
        x,y=[s[key][sub] if sub else s[key] for s in summaries]
        comparisons.append({'metric':name,'E01':x,'E03':y,'delta_E03_minus_E01':y-x,'relative_percent':(y-x)/x*100})
    result={'schema_version':1,'label':'E01 vs E03, one seed42 each; descriptive, no statistical significance',
            'delta_convention':'E03 minus E01; negative loss/BPB delta favors E03',
            'comparability_passed':True,'verified_runs':verified,'aggregate':aggregate,
            'same_training_settings':True,'same_dataset_tokenizer_code_environment':True,
            'only_model_factor_difference':'position','matched_validation':rows,'summary_comparison':comparisons,
            'limitations':['One seed; SD undefined','Same GPU model, separate sessions; no controlled speed repetitions',
                           'Validation loss and BPB mathematically related for matched character bytes',
                           'No test likelihood or cross-tokenizer ranking','No position extrapolation tested']}
    atomic_json(out/'comparison.json',result)
    for name,data in [('matched_validation',rows),('summary_comparison',comparisons),('verified_runs',verified),('aggregate',aggregate)]:
        with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for metric,ylabel in [('loss','Full sliding validation NLL (nats)'),('bpb','Conditional validation bits/byte')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(['E01 sinusoidal','E03 RoPE']):ax.plot([r['step'] for r in values[i]],[r['validation'][metric] for r in values[i]],marker='.',label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Matched seed42 positional ablation; observed points')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+metric+'.png'),dpi=200);plt.close(fig)
    for name,key,ylabel,smooth in [('training','training_loss','Training NLL (100-step trailing mean)',True),('throughput','tokens_per_second','Cumulative targets / training second',False)]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(['E01 sinusoidal','E03 RoPE']):
            h=histories[i];v=[r[key] for r in h]
            if smooth:v=np.convolve(v,np.ones(100)/100,'valid');steps=[r['step'] for r in h][99:]
            else:steps=[r['step'] for r in h]
            ax.plot(steps,v,label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Measured single-seed runs; separate GPU sessions')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+name+'.png'),dpi=200);plt.close(fig)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--rope',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.baseline,a.rope,a.root,a.output)
