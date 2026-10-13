"""Three verified seed42 width runs; no training or inferred missing results."""
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


def compare(root,artifacts,out):
    out.mkdir(parents=True,exist_ok=False)
    ids=['E13','E01','E14'];widths=[144,192,240];counts=[1021824,1804800,2808960]
    payloads=[];summaries=[];runs=[]
    for path,eid,width,count in zip(artifacts,ids,widths,counts):
        completion=json.loads((path/'completion.json').read_text())
        checkpoint=path/completion['final_checkpoint'];payload,_=read_checkpoint(checkpoint)
        m=payload['metadata'];assert m['resolved']['experiment_id']==eid and m['parameters']==count
        assert payload['config']['d_model']==width and payload['step']==5000
        payloads.append(payload);runs.append(checkpoint.parent.parent)
        summaries.append(json.loads((path/'analysis/summary.json').read_text()))
    verified,aggregate=collect(runs,root,5000,require_complete=False)
    baseline=payloads[1]['metadata']
    for p in payloads:
        m=p['metadata'];assert m['resolved']['training']==baseline['resolved']['training']
        assert m['resolved']['tokenizer_hash']==baseline['resolved']['tokenizer_hash']
        assert {k for k in p['config'] if p['config'][k]!=payloads[1]['config'][k]}<= {'d_model'}
        assert all(m['environment'][k]==baseline['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
        normalize=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
        assert normalize(m['environment'])==normalize(baseline['environment'])
    history=[p['training_state']['history'] for p in payloads]
    validations=[[h for h in rows if 'validation' in h] for rows in history]
    assert all([v['step'] for v in rows]==list(range(250,5001,250)) for rows in validations)
    rows=[]
    for values in zip(*validations):
        row={'step':values[0]['step']}
        for eid,v in zip(ids,values):
            assert v['validation']['scored_bytes']==55514 and v['validation']['context_length']==128
            row.update({eid+'_loss':v['validation']['loss'],eid+'_bpb':v['validation']['bpb'],eid+'_training_seconds':v['train_seconds']})
        rows.append(row)
    resources=[]
    for eid,width,s,h in zip(ids,widths,summaries,history):
        norms=np.array([v['gradient_norm'] for v in h])
        resources.append({'experiment':eid,'width':width,'head_dimension':width//6,'MLP_hidden':width*4,
            'parameters':s['parameters'],'best_step':s['best_validation']['step'],'best_loss':s['best_validation']['loss'],
            'best_bpb':s['best_validation']['bpb'],'final_loss':s['final_validation']['loss'],'final_bpb':s['final_validation']['bpb'],
            'training_seconds':s['training_seconds'],'validation_seconds':s['validation_seconds'],
            'peak_gpu_MiB':s['peak_gpu_bytes']/2**20,'updates_per_second':5000/s['training_seconds'],
            'tokens_per_second':s['training_tokens']/s['training_seconds'],'gradient_median':float(np.median(norms)),
            'gradient_p95':float(np.percentile(norms,95)),'clipped_updates':int((norms>1).sum())})
    atomic_json(out/'width_comparison.json',{'label':'Measured E13/E01/E14 seed42; single seed, SD undefined',
        'verified_runs':verified,'aggregate':aggregate,'resources':resources,'matched_validation':rows,
        'limitations':['Width, capacity, head dimension, MLP width and initialization change together',
                      'Separate T4 sessions; timing differences are observational','No test selection or significance claim']})
    for name,values in [('width_resources',resources),('width_matched_validation',rows)]:
        with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=220);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    for metric in ['loss','bpb']:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for eid,width,vs in zip(ids,widths,validations):ax.plot([v['step'] for v in vs],[v['validation'][metric] for v in vs],'.-',label=f'{eid}: width{width}')
        ax.set(xlabel='Completed optimizer updates',ylabel='Conditional BPB' if metric=='bpb' else 'Validation NLL (nats)',title='Full sliding validation; seed42');ax.legend();ax.grid(alpha=.2);save(fig,'width_validation_'+metric)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for eid,width,vs in zip(ids,widths,validations):ax.plot([v['train_seconds'] for v in vs],[v['validation']['bpb'] for v in vs],'.-',label=f'{eid}: width{width}')
    ax.set(xlabel='Measured training-only seconds',ylabel='Conditional validation BPB',title='Observed quality versus training cost');ax.legend();ax.grid(alpha=.2);save(fig,'width_bpb_training_time')
    fig,ax=plt.subplots(figsize=(7,4.5))
    for row in resources:
        ax.scatter(row['parameters']/1e6,row['best_bpb']);ax.annotate(row['experiment'],(row['parameters']/1e6,row['best_bpb']))
    ax.set(xlabel='Trainable parameters (millions)',ylabel='Best observed conditional BPB',title='Single-seed capacity/quality tradeoff');ax.grid(alpha=.2);save(fig,'width_bpb_parameters')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,label in zip(axes,['training_seconds','validation_seconds','peak_gpu_MiB'],['Training-only seconds','Full-validation seconds','Peak allocated GPU MiB']):ax.bar(ids,[v[key] for v in resources]);ax.set(ylabel=label)
    fig.suptitle('Measured resource costs; separate T4 sessions');save(fig,'width_resources')
    print(json.dumps(resources,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--e13',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--e14',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.root,[a.e13,a.baseline,a.e14],a.output)
