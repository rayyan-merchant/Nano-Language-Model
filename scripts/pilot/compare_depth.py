"""Generate a validated, single-seed E01/E08 comparison from genuine checkpoints."""
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
import torch
from nano_lm.artifacts import atomic_json
from nano_lm.checkpoint import read_checkpoint
from nano_lm.results import collect


def compare(baseline, six_layer, root, out):
    out.mkdir(parents=True,exist_ok=False)
    runs=[]; payloads=[]
    for artifact,eid in [(baseline,'E01'),(six_layer,'E08')]:
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
    assert a['parameters']==1804800 and b['parameters']==2694528
    ma,mb=a['resolved']['model'],b['resolved']['model']
    assert {k for k in ma if ma[k]!=mb[k]}=={'n_layers'} and ma['n_layers']==4 and mb['n_layers']==6
    freeze=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
    assert freeze(a['environment'])==freeze(b['environment'])
    assert all(a['environment'][k]==b['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
    assert torch.equal(payloads[0]['rng']['sampling'],payloads[1]['rng']['sampling'])
    assert np.array_equal(payloads[0]['training_state']['seen_bytes'],payloads[1]['training_state']['seen_bytes'])
    histories=[p['training_state']['history'] for p in payloads]
    values=[[r for r in h if 'validation' in r] for h in histories]
    assert all([r['step'] for r in v]==list(range(250,5001,250)) for v in values)
    rows=[]
    for av,bv in zip(*values):
        row={'step':av['step']}
        for metric in ['loss','bpb','perplexity']:
            x,y=av['validation'][metric],bv['validation'][metric]
            row.update({f'E01_{metric}':x,f'E08_{metric}':y,f'delta_{metric}':y-x,f'relative_{metric}_percent':(y-x)/x*100})
        rows.append(row)
    summaries=[json.loads((p/'analysis/summary.json').read_text()) for p in [baseline,six_layer]]
    thresholds=[]
    for reference_step in [1000,2000,3000,4000,5000]:
        reference=next(r for r in values[0] if r['step']==reference_step)['validation']['loss']
        crossing=next((r for r in values[1] if r['validation']['loss']<=reference),None)
        thresholds.append({'E01_reference_step':reference_step,'E01_reference_loss':reference,
                           'E08_first_observed_at_or_below_step':crossing['step'] if crossing else None,
                           'E08_training_seconds_at_that_observation':crossing['train_seconds'] if crossing else None,
                           'interpretation':'Retrospective recorded-point threshold; not interpolated crossing or early stopping'})
    comparisons=[]
    for name,key,sub in [('best_loss','best_validation','loss'),('final_loss','final_validation','loss'),('best_bpb','best_validation','bpb'),('final_bpb','final_validation','bpb'),('final_training_loss','final_training_loss',None),('training_seconds','training_seconds',None),('validation_seconds','validation_seconds',None),('peak_gpu_bytes','peak_gpu_bytes',None),('parameters','parameters',None)]:
        x,y=[s[key][sub] if sub else s[key] for s in summaries]
        comparisons.append({'metric':name,'E01':x,'E08':y,'delta_E08_minus_E01':y-x,'relative_percent':(y-x)/x*100})
    result={'schema_version':1,'label':'E01 vs E08, one seed42 each; descriptive, no statistical significance',
            'delta_convention':'E08 minus E01; negative loss/BPB delta favors E08',
            'comparability_passed':True,'verified_runs':verified,'aggregate':aggregate,
            'same_training_settings':True,'same_dataset_tokenizer_code_environment':True,
            'same_final_sampling_RNG_and_byte_coverage':True,
            'only_model_factor_difference':'n_layers','matched_validation':rows,'summary_comparison':comparisons,
            'retrospective_threshold_observations':thresholds,
            'limitations':['One seed; SD undefined','Same GPU model, separate sessions; no controlled speed repetitions',
                           'Validation loss and BPB mathematically related for matched character bytes',
                           'No test likelihood or cross-tokenizer ranking','Depth increases capacity by889728 parameters; equal tokens/updates do not equal compute','E09 not run; all its metrics pending']}
    atomic_json(out/'comparison.json',result)
    depth_framework=[]
    for eid,layers,summary in [('E01',4,summaries[0]),('E08',6,summaries[1]),('E09',8,None)]:
        depth_framework.append({'experiment':eid,'layers':layers,'seed':42,
                                'status':'observed' if summary else 'NOT_AUTHORIZED_NOT_RUN',
                                'parameters':summary['parameters'] if summary else None,
                                'best_bpb':summary['best_validation']['bpb'] if summary else None,
                                'final_bpb':summary['final_validation']['bpb'] if summary else None,
                                'training_seconds':summary['training_seconds'] if summary else None})
    atomic_json(out/'depth_framework.json',{'scope':'Depth also changes capacity/compute; E09 metrics intentionally absent','rows':depth_framework})
    gradient_rows=[]
    for eid,h in zip(['E01','E08'],histories):
        norms=np.array([r['gradient_norm'] for r in h])
        assert np.isfinite(norms).all()
        gradient_rows.append({'experiment':eid,'scope':'Recorded pre-clipping aggregate norms, not per-layer research gradients',
                              'maximum':float(norms.max()),'median':float(np.median(norms)),
                              'p95':float(np.percentile(norms,95)),
                              'clipped_updates':int((norms>1).sum()),
                              'first100_mean':float(norms[:100].mean()),'last100_mean':float(norms[-100:].mean())})
    atomic_json(out/'gradient_comparison.json',{'scope':'Descriptive within-run norm statistics; not across-seed uncertainty','rows':gradient_rows})
    for metric,ylabel in [('training_seconds','Training-only seconds'),('peak_gpu_bytes','Peak allocated memory (MiB)')]:
        fig,ax=plt.subplots(figsize=(7,4.5));data=[s[metric] for s in summaries]
        if metric=='peak_gpu_bytes':data=[v/2**20 for v in data]
        ax.bar(['E01:4 blocks','E08:6 blocks'],data,color=['tab:blue','tab:orange'])
        ax.set(ylabel=ylabel,title='Measured seed42 runs; separate T4 sessions')
        fig.tight_layout();fig.savefig(out/('comparison_'+metric+'.png'),dpi=200);plt.close(fig)
    for name,data in [('matched_validation',rows),('summary_comparison',comparisons),('verified_runs',verified),('aggregate',aggregate)]:
        with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for metric,ylabel in [('loss','Full sliding validation NLL (nats)'),('bpb','Conditional validation bits/byte')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(['E01 four blocks','E08 six blocks']):ax.plot([r['step'] for r in values[i]],[r['validation'][metric] for r in values[i]],marker='.',label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Matched seed42 depth ablation; observed points')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+metric+'.png'),dpi=200);plt.close(fig)
    for name,key,ylabel,smooth in [('training','training_loss','Training NLL (100-step trailing mean)',True),('throughput','tokens_per_second','Cumulative targets / training second',False)]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(['E01 four blocks','E08 six blocks']):
            h=histories[i];v=[r[key] for r in h]
            if smooth:v=np.convolve(v,np.ones(100)/100,'valid');steps=[r['step'] for r in h][99:]
            else:steps=[r['step'] for r in h]
            ax.plot(steps,v,label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Measured single-seed runs; separate GPU sessions')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+name+'.png'),dpi=200);plt.close(fig)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--six-layer',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.baseline,a.six_layer,a.root,a.output)
