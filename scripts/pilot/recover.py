"""Fresh CUDA restore audit of saved pilot state, with no optimizer updates."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint, restore_training, save_checkpoint, load_for_inference
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.generate import generate
from nano_lm.storage import restore_snapshot, verify_snapshot
from nano_lm.train import environment
from nano_lm.verification import equal_nested


def recover(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    completions=list(Path('/kaggle/input').rglob('completion.json'));assert len(completions)==1
    original=completions[0].parent;completion=json.loads(completions[0].read_text())
    snapshots=[]
    for path in (original/'snapshots').iterdir():
        manifest=verify_snapshot(path)
        if manifest['step']==5000:snapshots.append((path,manifest))
    snapshot,manifest=max(snapshots,key=lambda item:len(item[1]['files']))
    restored=restore_snapshot(snapshot,out/'restored');final=restored/'checkpoints/step_005000.pt'
    payload,tokenizer=read_checkpoint(final);env=environment(root)
    assert env['code_hash']==payload['metadata']['resolved']['code_hash'] and not env['git_dirty']
    assert env['python']=='3.11.14' and env['torch']=='2.5.1+cu124' and env['cuda']=='12.4' and env['gpu']=='Tesla T4'
    normalize=lambda lines:sorted(x for x in lines if not x.startswith('-e '))
    assert normalize(env['packages'])==normalize(payload['metadata']['environment']['packages'])
    model=GPT(ModelConfig(**payload['config'])).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    scaler=torch.amp.GradScaler('cuda',enabled=False);generator=torch.Generator()
    restore_training(payload,model,optimizer,scheduler,generator);scaler.load_state_dict(payload['scaler'] or {})
    roundtrip=out/'restored_state_step_005000.pt'
    save_checkpoint(roundtrip,model,tokenizer,optimizer,scheduler,generator,5000,payload['metadata'],scaler,payload['training_state'])
    actual,_=read_checkpoint(roundtrip)
    exact={key:equal_nested(actual[key],payload[key]) for key in ['model','optimizer','scheduler','scaler','rng','training_state']}
    assert all(exact.values())
    samples=[]
    for label,name in [('best',payload['training_state']['best_checkpoint']),('final','step_005000.pt')]:
        path=restored/'checkpoints'/name;loaded,tok=load_for_inference(path);loaded.to('cuda')
        ids=generate(loaded,torch.tensor([tok.encode('ROMEO:\n')],device='cuda'),64,greedy=True,seed=42)
        samples.append({'policy':label,'checkpoint':name,'sha256':file_hash(path),'text':tok.decode(ids[0].tolist()),'device':'cuda','greedy':True,'seed':42,'max_new_tokens':64})
    atomic_json(out/'recovery.json',{'status':'fresh_saved_output_CUDA_state_restore_passed_no_updates','source_commit':env['git_commit'],'trained_source_commit':payload['metadata']['environment']['git_commit'],'environment':env,'snapshot_hash':manifest['hash'],'run_id':completion['run_id'],'exact_state':exact,'optimizer_updates_before':5000,'optimizer_updates_after':5000,'additional_optimizer_updates':0,'samples':samples,'limit':'State restoration and roundtrip verified; actual 10k extension was not started'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();recover(a.root,a.output)
