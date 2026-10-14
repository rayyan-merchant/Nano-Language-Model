"""Create pilot summaries, figures and all fixed-protocol samples from real weights."""
import argparse
import csv
import json
from pathlib import Path
import time
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nano_lm.artifacts import atomic_json, file_hash, digest
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate


def analyze(root,run,out,device,skip_generation=False,development_check=False):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    # F01 is separately authorized; historical E01-E14 targets remain 5000.
    metadata=json.loads((run/'metadata.json').read_text())
    target=10000 if metadata['resolved']['experiment_id']=='F01' else 5000
    checkpoint_path = (max((run/'checkpoints').glob('step_*.pt'))
                       if development_check else run/f'checkpoints/step_{target:06d}.pt')
    payload,_=read_checkpoint(checkpoint_path)
    experiment=payload['metadata']['resolved']['experiment_id']
    seed=payload['metadata']['resolved']['training']['seed']
    label=f'{experiment} seed{seed}: '+('DEVELOPMENT postprocessing check' if development_check else 'single-seed research pilot')
    state=payload['training_state'];history=state['history'];val=[r for r in history if 'validation' in r]
    if not development_check:assert len(history)==target and len(val)==target//250
    else:assert payload['metadata']['resolved']['training']['run_kind']=='smoke' and val
    best=min(val,key=lambda r:r['validation']['loss']);last=history[-1]
    if not development_check:assert state['best_checkpoint']==f"step_{best['step']:06d}.pt"
    atomic_json(out/'measured_history.json',{'label':label,'training':history,'validation':val})
    with (out/'validation.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['step','loss','perplexity','bpb','generalization_gap']);writer.writeheader()
        for r in val:writer.writerow({'step':r['step'],**{k:r['validation'][k] for k in ['loss','perplexity','bpb']},'generalization_gap':r['generalization_gap']})
    figures=out/'figures';figures.mkdir()
    steps=np.array([r['step'] for r in history]);loss=np.array([r['training_loss'] for r in history]);vs=[r['step'] for r in val]
    def figure(name,ylabel,series):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for x,y,series_label in series:ax.plot(x,y,label=series_label,linewidth=1.3)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title=label)
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=200);fig.savefig(figures/(name+'.pdf'));plt.close(fig)
    window=min(100,len(loss));smooth=np.convolve(loss,np.ones(window)/window,mode='valid')
    figure('training_loss','Token NLL (nats)',[(steps,loss,'Observed step loss'),(steps[window-1:],smooth,f'{window}-step trailing mean')])
    vl=[r['validation']['loss'] for r in val]
    figure('validation_loss','Full sliding validation NLL (nats)',[(vs,vl,'Measured validation points')])
    figure('combined_loss','Token NLL (nats)',[(steps[window-1:],smooth,f'Training: {window}-step trailing mean'),(vs,vl,'Development capped sliding validation' if development_check else 'Full sliding validation')])
    figure('validation_bpb','Conditional validation bits per source byte',[(vs,[r['validation']['bpb'] for r in val],f"{val[0]['validation']['scored_bytes']:,} scored bytes; {val[0]['validation']['conditioning_bytes']}-byte conditioning")])
    figure('generalization_gap','Validation NLL minus training NLL',[(vs,[r['generalization_gap'] for r in val],'Validation minus same-step batch loss')])
    figure('throughput','Cumulative predicted tokens / training second',[(steps,[r['tokens_per_second'] for r in history],'Training throughput; excludes validation/IO')])
    figure('training_time','Accumulated training seconds',[(steps,[r['train_seconds'] for r in history],'Measured training time')])
    figure('gradient_norm','Gradient norm before clipping',[(steps,[r['gradient_norm'] for r in history],'Observed norm; clipping threshold 1.0')])
    figure('learning_rate','Learning rate',[(steps,[r['lr'] for r in history],'Constant approved learning rate')])
    final_metrics=val[-1]['validation'];late=[r for r in val if r['step']>=3000]
    summary={'label':label,'run_id':payload['metadata']['run_id'],'source_commit':payload['metadata']['environment']['git_commit'],'code_hash':payload['metadata']['resolved']['code_hash'],'config_hash':payload['metadata']['config_hash'],'parameters':payload['metadata']['parameters'],'completed_optimizer_updates':payload['step'],'initial_recorded_training':{'step':1,'loss':loss[0].item()},'initial_recorded_validation':{'step':val[0]['step'],**val[0]['validation']},'final_training_loss':last['training_loss'],'final_training_trailing_100_loss':float(smooth[-1]),'final_validation':final_metrics,'best_validation':{'step':best['step'],**best['validation']},'milestones':{str(r['step']):r['validation'] for r in val if r['step']%1000==0},'training_seconds':state['train_seconds'],'validation_seconds':state['eval_seconds'],'peak_gpu_bytes':state['peak_gpu_bytes'],'training_tokens':state['training_tokens'],'training_source_bytes':state['training_bytes'],'unique_target_bytes':last['unique_target_bytes'],'late_validation_slope_per_1000_steps':float(np.polyfit([r['step'] for r in late],[r['validation']['loss'] for r in late],1)[0]*1000) if len(late)>1 else None,'loss_improvement_3000_to_5000':next(r['validation']['loss'] for r in val if r['step']==3000)-final_metrics['loss'] if not development_check else None,'train_loss_trailing_windows':{str(s):float(np.mean(loss[s-100:s])) for s in [1000,2000,3000,4000,5000] if s<=len(loss)},'uncertainty':'One seed; no SD or significance estimate; full study remains NO-GO'}
    if experiment=='F01' and not development_check:
        summary['loss_improvement_3000_to_5000']=next(r['validation']['loss'] for r in val if r['step']==3000)-next(r['validation']['loss'] for r in val if r['step']==5000)
        summary['loss_improvement_5000_to_10000']=next(r['validation']['loss'] for r in val if r['step']==5000)-final_metrics['loss']
        summary['late_validation_slope_6000_to_10000']=float(np.polyfit([r['step'] for r in val if r['step']>=6000],[r['validation']['bpb'] for r in val if r['step']>=6000],1)[0]*1000)
        summary['train_loss_trailing_windows']={str(s):float(np.mean(loss[s-100:s])) for s in range(1000,10001,1000)}
        summary['uncertainty']='One exploratory combined architecture, one seed; no variance or additive-effect claim'
    atomic_json(out/'summary.json',summary)
    if skip_generation:
        print(json.dumps(summary,indent=2),flush=True)
        return
    protocol=json.loads((root/'configs/generation_protocol.json').read_text());samples=[];began=time.perf_counter()
    if development_check:protocol={**protocol,'max_new_tokens':2,'sampling_seeds':[42],'temperatures':[0.7]}
    for checkpoint_policy,name in ([('development',checkpoint_path.name)] if development_check else [('best',state['best_checkpoint']),('final',f'step_{target:06d}.pt')]):
        checkpoint=run/'checkpoints'/name;model,tokenizer=load_for_inference(checkpoint);model.to(device)
        for prompt in protocol['prompts']:
            settings=[{'greedy':True,'temperature':1.0,'seed':protocol['sampling_seeds'][0]}] if protocol['greedy'] else []
            settings += [{'greedy':False,'temperature':temp,'seed':seed} for temp in protocol['temperatures'] for seed in protocol['sampling_seeds']]
            for setting in settings:
                ids=torch.tensor([tokenizer.encode(prompt)],device=device)
                result=generate(model,ids,protocol['max_new_tokens'],**setting)
                samples.append({'checkpoint_policy':checkpoint_policy,'checkpoint':name,'checkpoint_sha256':file_hash(checkpoint),'prompt':prompt,'max_new_tokens':protocol['max_new_tokens'],**setting,'generation_device':device,'text':tokenizer.decode(result[0].tolist())})
        del model;torch.cuda.empty_cache() if device=='cuda' else None
    atomic_json(out/'generation.json',{'protocol':protocol,'protocol_sha256':digest(protocol) if development_check else file_hash(root/'configs/generation_protocol.json'),'protocol_source_sha256':file_hash(root/'configs/generation_protocol.json'),'development_check':development_check,'samples':samples,'seconds':time.perf_counter()-began,'selection':'Entire fixed grid; no cherry-picking; best and final separately labeled'})
    with (out/'generation.md').open('x',encoding='utf-8') as f:
        f.write(f'# Genuine {experiment} seed{seed} model generations\n\n'+('DEVELOPMENT: shortened generation plumbing check, not research.\n' if development_check else 'All fixed prompts/settings included. No editing of generated text.\n'))
        for i,s in enumerate(samples):f.write(f"\n## {i+1}: {s['checkpoint_policy']} {s['checkpoint']}; greedy={s['greedy']}, temperature={s['temperature']}, seed={s['seed']}\n\nPrompt: {s['prompt']!r}\n\n```text\n{s['text']}\n```\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--skip-generation',action='store_true');p.add_argument('--development-check',action='store_true')
    a=p.parse_args();analyze(a.root,a.run,a.output,a.device,a.skip_generation,a.development_check)
