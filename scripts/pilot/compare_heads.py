"""Validate genuine E01/E06 or E01/E10 checkpoints and resource tradeoffs."""
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


def compare(baseline, four_head, root, out, experiment="E06"):
    if experiment not in {"E06","E10","E11","E13"}:raise ValueError("Unsupported comparison")
    factor="d_model" if experiment=="E13" else "context_length" if experiment=="E11" else "n_heads" if experiment=="E06" else "mlp_expansion"
    sizes=(1804800,1021824) if experiment=="E13" else (1804800,1804800) if experiment in {"E06","E11"} else (1804800,1213440)
    labels=["E01: baseline", "E13: width144" if experiment=="E13" else "E11: context64" if experiment=="E11" else "E06: four heads" if experiment=="E06" else "E10: 2x MLP"]
    out.mkdir(parents=True,exist_ok=False)
    runs=[]; payloads=[]
    for artifact,eid in [(baseline,'E01'),(four_head,experiment)]:
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
    assert (a['parameters'],b['parameters'])==sizes
    ma,mb=a['resolved']['model'],b['resolved']['model']
    assert {k for k in ma if ma[k]!=mb[k]}=={factor}
    assert ma[factor]==(192 if experiment=='E13' else 128 if experiment=='E11' else 6 if experiment=='E06' else 4) and mb[factor]==(144 if experiment=='E13' else 64 if experiment=='E11' else 4 if experiment=='E06' else 2)
    freeze=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
    assert freeze(a['environment'])==freeze(b['environment'])
    assert all(a['environment'][k]==b['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
    sampling=[]
    if experiment=='E11':
        from sampling_audit import replay
        sampling=[replay(p) for p in payloads]
    else:
        assert torch.equal(payloads[0]['rng']['sampling'],payloads[1]['rng']['sampling'])
        assert np.array_equal(payloads[0]['training_state']['seen_bytes'],payloads[1]['training_state']['seen_bytes'])
    histories=[p['training_state']['history'] for p in payloads]
    values=[[r for r in h if 'validation' in r] for h in histories]
    assert all([r['step'] for r in v]==list(range(250,5001,250)) for v in values)
    rows=[]
    for av,bv in zip(*values):
        row={'step':av['step']}
        if experiment in {'E10','E11','E13'}:row.update({'E01_training_seconds':av['train_seconds'],experiment+'_training_seconds':bv['train_seconds']})
        for metric in ['loss','bpb','perplexity']:
            x,y=av['validation'][metric],bv['validation'][metric]
            row.update({f'E01_{metric}':x,f'{experiment}_{metric}':y,f'delta_{metric}':y-x,f'relative_{metric}_percent':(y-x)/x*100})
        rows.append(row)
    summaries=[json.loads((p/'analysis/summary.json').read_text()) for p in [baseline,four_head]]
    thresholds=[]
    for reference_step in [1000,2000,3000,4000,5000]:
        reference=next(r for r in values[0] if r['step']==reference_step)['validation']['loss']
        crossing=next((r for r in values[1] if r['validation']['loss']<=reference),None)
        thresholds.append({'E01_reference_step':reference_step,'E01_reference_loss':reference,
                           f'{experiment}_first_observed_at_or_below_step':crossing['step'] if crossing else None,
                           f'{experiment}_training_seconds_at_that_observation':crossing['train_seconds'] if crossing else None,
                           'interpretation':'Retrospective recorded-point threshold; not interpolated crossing or early stopping'})
    comparisons=[]
    for name,key,sub in [('best_loss','best_validation','loss'),('final_loss','final_validation','loss'),('best_bpb','best_validation','bpb'),('final_bpb','final_validation','bpb'),('final_training_loss','final_training_loss',None),('training_seconds','training_seconds',None),('validation_seconds','validation_seconds',None),('peak_gpu_bytes','peak_gpu_bytes',None),('parameters','parameters',None)]:
        x,y=[s[key][sub] if sub else s[key] for s in summaries]
        comparisons.append({'metric':name,'E01':x,experiment:y,f'delta_{experiment}_minus_E01':y-x,'relative_percent':(y-x)/x*100})
    result={'schema_version':1,'label':f'E01 vs {experiment}, one seed42 each; descriptive, no statistical significance',
            'delta_convention':f'{experiment} minus E01; negative loss/BPB delta favors {experiment}',
            'comparability_passed':True,'verified_runs':verified,'aggregate':aggregate,
            'same_training_settings':True,'same_dataset_tokenizer_code_environment':True,
            'same_final_sampling_RNG_and_byte_coverage':experiment!='E11','sampling_audits':sampling,
            'only_model_factor_difference':factor,'matched_validation':rows,'summary_comparison':comparisons,
            'retrospective_threshold_observations':thresholds,
            'limitations':['One seed; SD undefined','Same GPU model, separate sessions; no controlled speed repetitions',
                           'Validation loss and BPB mathematically related for matched character bytes',
                           'No test likelihood or cross-tokenizer ranking','No other widths/depths tested']}
    if experiment in {'E10','E11','E13'}:
        x,y=summaries
        result['gradient_statistics']=[]
        for eid,h in zip(['E01',experiment],histories):
            norms=np.array([r['gradient_norm'] for r in h]);assert np.isfinite(norms).all()
            result['gradient_statistics'].append({'experiment':eid,'median':float(np.median(norms)),'p95':float(np.percentile(norms,95)),'maximum':float(norms.max()),'clipped_updates':int((norms>1).sum()),'clipping_fraction':float((norms>1).mean()),'first100_mean':float(norms[:100].mean()),'last100_mean':float(norms[-100:].mean()),'scope':'Aggregate pre-clipping research norms; no longitudinal per-layer gradient data'})
        if experiment=='E10':
            result['efficiency_tradeoffs']={'parameters_removed':591360,'parameter_reduction_percent':100*591360/1804800,'delta_best_BPB_per_million_parameters_removed':(y['best_validation']['bpb']-x['best_validation']['bpb'])/(591360/1e6),'E01_targets_per_training_second':x['training_tokens']/x['training_seconds'],'E10_targets_per_training_second':y['training_tokens']/y['training_seconds'],'training_seconds_saved':x['training_seconds']-y['training_seconds'],'GPU_allocated_bytes_saved':x['peak_gpu_bytes']-y['peak_gpu_bytes'],'scope':'Descriptive quality-resource tradeoffs; BPB divided by parameters is not an intrinsic quality/efficiency score. Capacity and MLP architecture change together.'}
    if experiment=='E11':
        result['limitations'] += ['64 vs128 conditioning tokens; common256-byte exclusion is not context length','64 vs32 sampled windows/update; equal4096 targets do not imply identical compute or sampler/dropout progression']
        atomic_json(out/'context_study_framework.json',{'E11':summaries[1],'E01':summaries[0],'E12':None,'E12_status':'NOT_AUTHORIZED_NOT_RUN'})
    if experiment=='E13':
        result['width_capacity_tradeoff']={'parameters_removed':782976,'reduction_percent':100*782976/1804800,'head_dimensions':[32,24],'MLP_hidden_dimensions':[768,576],'scope':'Width changes capacity/head dimension/MLP width together; no isolated causal claim'}
        atomic_json(out/'width_study_framework.json',{'E13':summaries[1],'E01':summaries[0],'E14':None,'E14_status':'NOT_AUTHORIZED_NOT_RUN'})
    atomic_json(out/'comparison.json',result)
    for name,data in [('matched_validation',rows),('summary_comparison',comparisons),('verified_runs',verified),('aggregate',aggregate)]:
        with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for metric,ylabel in [('loss','Full sliding validation NLL (nats)'),('bpb','Conditional validation bits/byte')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(labels):ax.plot([r['step'] for r in values[i]],[r['validation'][metric] for r in values[i]],marker='.',label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title=f'Matched seed42 {factor} ablation; observed points')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+metric+'.png'),dpi=200)
        if experiment in {'E10','E11','E13'}:fig.savefig(out/('comparison_'+metric+'.pdf'))
        plt.close(fig)
    for name,key,ylabel,smooth in [('training','training_loss','Training NLL (100-step trailing mean)',True),('throughput','tokens_per_second','Cumulative targets / training second',False)]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for i,eid in enumerate(labels):
            h=histories[i];v=[r[key] for r in h]
            if smooth:v=np.convolve(v,np.ones(100)/100,'valid');steps=[r['step'] for r in h][99:]
            else:steps=[r['step'] for r in h]
            ax.plot(steps,v,label=eid)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Measured single-seed runs; separate GPU sessions')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/('comparison_'+name+'.png'),dpi=200)
        if experiment in {'E10','E11','E13'}:fig.savefig(out/('comparison_'+name+'.pdf'))
        plt.close(fig)
    if experiment in {'E10','E11','E13'}:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for label,v in zip(labels,values):ax.plot([r['train_seconds'] for r in v],[r['validation']['bpb'] for r in v],marker='.',label=label)
        ax.set(xlabel='Measured cumulative training-only seconds',ylabel='Conditional validation BPB',title='Observed seed42 quality/cost tradeoff; separate T4 sessions')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'comparison_bpb_training_time.png',dpi=200);fig.savefig(out/'comparison_bpb_training_time.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11,4))
        for ax,key,scale,label in zip(axes,['parameters','training_seconds','peak_gpu_bytes'],[1e6,1,2**20],['Parameters (millions)','Training-only seconds','Peak allocated GPU MiB']):
            ax.bar(['E01',experiment],[s[key]/scale for s in summaries]);ax.set(ylabel=label)
        fig.suptitle('Genuine single-seed resource measurements');fig.tight_layout();fig.savefig(out/'comparison_resources.png',dpi=200);fig.savefig(out/'comparison_resources.pdf');plt.close(fig)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--four-head','--candidate',dest='four_head',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment',choices=['E06','E10','E11','E13'],default='E06')
    a=p.parse_args();compare(a.baseline,a.four_head,a.root,a.output,a.experiment)
