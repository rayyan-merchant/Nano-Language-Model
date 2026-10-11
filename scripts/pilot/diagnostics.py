"""Audit real pilot histories and describe train-text overlap without test selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from nano_lm.checkpoint import read_checkpoint


def longest_overlap(text,training):
    low,high=0,len(text);example='';offset=None
    while low<high:
        size=(low+high+1)//2;found=False
        for i in range(len(text)-size+1):
            part=text[i:i+size];where=training.find(part)
            if where>=0:
                low=size;example=part;offset=where;found=True;break
        if not found:high=size-1
    return {'longest_exact_training_substring_characters':low,'substring':example,'training_offset':offset,'interpretation':'Exact phrase overlap is a memorization indicator, not proof; ordinary Shakespeare language repeats.'}


def diagnostics(artifacts,training_path,output):
    completion=json.loads((artifacts/'completion.json').read_text())
    run=artifacts/completion['final_checkpoint'];run=run.parent.parent
    final,_=read_checkpoint(run/'checkpoints/step_005000.pt');h=final['training_state']['history']
    manifest=json.loads((run/'dataset_manifest.json').read_text());raw=training_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['partitions']['train']['sha256']
    text=raw.decode('utf-8');checks=[]
    for path in sorted((run/'checkpoints').glob('step_*.pt')):
        payload,_=read_checkpoint(path);step=payload['step'];state=payload['training_state']
        assert payload['metadata']==final['metadata'] and len(state['history'])==step
        assert state['training_tokens']==step*4096 and state['training_bytes']==step*4096
        optimizer_steps=sorted(set(int(value['step'].item()) for value in payload['optimizer']['state'].values()))
        assert optimizer_steps==[step] and payload['scheduler']['last_epoch']==step
        checks.append({'file':path.name,'optimizer_step':step,'optimizer_parameter_state_steps':optimizer_steps,'scheduler_epoch':payload['scheduler']['last_epoch'],'history_length':len(state['history'])})
    assert len(checks)==20
    assert all(np.isfinite(r['training_loss']) and np.isfinite(r['gradient_norm']) for r in h)
    validations=[r for r in h if 'validation' in r]
    for r in validations:
        v=r['validation'];assert v['scored_bytes']==v['scored_tokens']==55514
        assert math.isclose(v['bpb'],v['loss']/math.log(2),rel_tol=1e-12)
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='step' for e in events)==5000
    assert not any(e['event'] in {'failure','resume'} for e in events)
    generation=json.loads((artifacts/'analysis/generation.json').read_text());samples=[]
    for s in generation['samples']:
        if s['checkpoint_policy']=='best' and (s['greedy'] or (s['temperature']==0.7 and s['seed']==42)):
            continuation=s['text'][len(s['prompt']):]
            ngrams=[continuation[i:i+8] for i in range(max(0,len(continuation)-7))]
            samples.append({'prompt':s['prompt'],'greedy':s['greedy'],'temperature':s['temperature'],'seed':s['seed'],'repeated_8gram_position_fraction':1-len(set(ngrams))/len(ngrams) if ngrams else 0,**longest_overlap(continuation,text)})
    intervals=[]
    for start,end in [(250,1000),(1000,2000),(2000,3000),(3000,4000),(4000,5000),(3000,5000)]:
        a=next(r for r in validations if r['step']==start)['validation']['loss'];b=next(r for r in validations if r['step']==end)['validation']['loss']
        intervals.append({'start':start,'end':end,'loss_decrease':a-b,'decrease_per_1000_steps':(a-b)*1000/(end-start)})
    value={'status':'all20_checkpoint_optimizer_and_scheduler_steps_verified','checkpoint_audit':checks,'journal_steps':5000,'failures':0,'resumes':0,'validation_BPB_analytic_character_identity_passed':True,'training_text_sha256':hashlib.sha256(raw).hexdigest(),'source_byte_coverage_fraction':h[-1]['unique_target_bytes']/len(raw),'late_validation_measurements':[{'step':r['step'],'loss':r['validation']['loss']} for r in validations if r['step']>=3000],'improvement_intervals':intervals,'generation_diagnostics':samples,'limitations':'No test likelihood/selection. Repetition and exact train-text phrases are descriptive indicators, not conclusive memorization/overfitting tests.'}
    with output.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    print(json.dumps({'status':value['status'],'intervals':intervals,'sample_count':len(samples)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--training-text',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnostics(a.artifacts,a.training_text,a.output)
