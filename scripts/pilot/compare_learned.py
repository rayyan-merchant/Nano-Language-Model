"""Validated three-way character-position comparison from genuine saved states."""
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


def compare(artifacts,root,out):
    out.mkdir(parents=True,exist_ok=False)
    ids=['E01','E03','E04'];positions=['sinusoidal','rope','learned']
    payloads=[];summaries=[];runs=[]
    for artifact,eid,position,count in zip(artifacts,ids,positions,[1804800,1804800,1829376]):
        completion=json.loads((artifact/'completion.json').read_text())
        checkpoint=artifact/completion['final_checkpoint'];p,_=read_checkpoint(checkpoint)
        assert p['step']==5000 and p['metadata']['resolved']['experiment_id']==eid
        assert p['metadata']['resolved']['model']['position']==position and p['metadata']['parameters']==count
        payloads.append(p);runs.append(checkpoint.parent.parent)
        summaries.append(json.loads((artifact/'analysis/summary.json').read_text()))
    verified,aggregate=collect(runs,root,5000,require_complete=False)
    a=payloads[0]['metadata'];normalize=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
    for p in payloads[1:]:
        b=p['metadata'];ma,mb=a['resolved']['model'],b['resolved']['model']
        assert {k for k in ma if ma[k]!=mb[k]}=={'position'}
        for key in ['training','dataset_hash','tokenizer_hash','tokenizer_kind','code_hash']:
            assert a['resolved'][key]==b['resolved'][key],key
        assert normalize(a['environment'])==normalize(b['environment'])
        assert all(a['environment'][k]==b['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
        assert torch.equal(payloads[0]['rng']['sampling'],p['rng']['sampling'])
        assert np.array_equal(payloads[0]['training_state']['seen_bytes'],p['training_state']['seen_bytes'])
    histories=[p['training_state']['history'] for p in payloads]
    values=[[r for r in h if 'validation' in r] for h in histories]
    assert all([r['step'] for r in v]==list(range(250,5001,250)) for v in values)
    rows=[]
    for index in range(20):
        row={'step':values[0][index]['step']}
        for eid,v in zip(ids,values):
            for key in ['loss','bpb','perplexity']:row[f'{eid}_{key}']=v[index]['validation'][key]
        for ref in ['E01','E03']:
            for key in ['loss','bpb']:
                x,y=row[f'{ref}_{key}'],row[f'E04_{key}']
                row[f'E04_minus_{ref}_{key}']=y-x;row[f'E04_relative_to_{ref}_{key}_percent']=(y-x)/x*100
        rows.append(row)
    summary_rows=[]
    for name,key,sub in [('best_loss','best_validation','loss'),('final_loss','final_validation','loss'),('best_bpb','best_validation','bpb'),('final_bpb','final_validation','bpb'),('final_training_loss','final_training_loss',None),('training_seconds','training_seconds',None),('validation_seconds','validation_seconds',None),('peak_gpu_bytes','peak_gpu_bytes',None),('parameters','parameters',None)]:
        row={'metric':name}
        for eid,s in zip(ids,summaries):row[eid]=s[key][sub] if sub else s[key]
        for ref in ['E01','E03']:
            row[f'E04_minus_{ref}']=row['E04']-row[ref]
            row[f'E04_relative_to_{ref}_percent']=(row['E04']-row[ref])/row[ref]*100
        summary_rows.append(row)
    # E02 values are genuine committed evidence, contextual BPB only, not token NLL ranking.
    e02_path=root/'report/pilots/E02_seed42/summary.json'
    e02=json.loads(e02_path.read_text()) if e02_path.exists() else None
    result={'schema_version':1,'label':'E01/E03/E04 seed42 character-position comparison; one seed each',
            'comparability_passed':True,'verified_runs':verified,'aggregate':aggregate,
            'same_dataset_tokenizer_code_environment':True,'same_final_sampling_RNG_and_byte_coverage':True,
            'matched_validation':rows,'summary_comparison':summary_rows,'E02_contextual_summary':e02,
            'limitations':['One seed; across-seed SD undefined','E04 adds24576 parameters and initialization RNG consumption changes later weights',
                          'Separate T4 sessions; no repeated controlled speed benchmark','E02 has BPE/different byte context,capacity,source exposure; do not rank token NLL/perplexity',
                          'No test likelihood or extrapolation evaluation']}
    atomic_json(out/'comparison.json',result)
    for name,data in [('matched_validation',rows),('summary_comparison',summary_rows),('verified_runs',verified),('aggregate',aggregate)]:
        with (out/(name+'.csv')).open('x',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    overall=[]
    indexed=dict(zip(ids,summaries))
    if e02:indexed['E02']=e02
    for eid in ['E01','E02','E03','E04']:
        if eid not in indexed:continue
        s=indexed[eid]
        overall.append({'experiment_id':eid,'seed':42,'tokenizer':'BPE' if eid=='E02' else 'character',
                        'parameters':s['parameters'],'completed_updates':s['completed_optimizer_updates'],
                        'best_step':s['best_validation']['step'],'best_conditional_bpb':s['best_validation']['bpb'],
                        'final_conditional_bpb':s['final_validation']['bpb'],'training_predicted_tokens':s['training_tokens'],
                        'training_source_byte_exposure':s['training_source_bytes'],
                        'comparison_scope':'Contextual BPE comparison;capacity/byte-context/exposure differ' if eid=='E02' else 'Character positional comparison;E04 adds capacity and changes initialization RNG consumption'})
    with (out/'overall_bpb.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(overall[0]));w.writeheader();w.writerows(overall)
    for metric,ylabel in [('loss','Full sliding validation NLL (nats)'),('bpb','Conditional validation bits/source byte')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for eid,pos,v in zip(ids,positions,values):ax.plot([r['step'] for r in v],[r['validation'][metric] for r in v],marker='.',label=f'{eid} {pos}')
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Character positional modes: measured seed42 curves')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'comparison_{metric}.png',dpi=200);plt.close(fig)
    for name,key,ylabel,smooth in [('training','training_loss','Training NLL (100-step trailing mean)',True),('throughput','tokens_per_second','Cumulative targets/training second',False)]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for eid,pos,h in zip(ids,positions,histories):
            v=[r[key] for r in h];steps=[r['step'] for r in h]
            if smooth:v=np.convolve(v,np.ones(100)/100,'valid');steps=steps[99:]
            ax.plot(steps,v,label=f'{eid} {pos}')
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='Single-seed measurements; separate T4 sessions')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'comparison_{name}.png',dpi=200);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--rope',type=Path,required=True);p.add_argument('--learned',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare([a.baseline,a.rope,a.learned],a.root,a.output)
