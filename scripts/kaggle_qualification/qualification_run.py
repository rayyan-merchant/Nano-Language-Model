"""Thin qualification orchestration, never a research run or a second trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.bpe import from_artifact
from nano_lm.checkpoint import load_for_inference, read_checkpoint
from nano_lm.evaluate import evaluate
from nano_lm.generate import generate
from nano_lm.prepare import prepare, validate
from nano_lm.storage import export_run, restore_snapshot, verify_snapshot
from nano_lm.train import TrainSettings, environment, train
from nano_lm.verification import verify, equal_nested


def command(args, root, output):
    began=time.perf_counter()
    result=subprocess.run(args,cwd=root,capture_output=True,text=True)
    atomic_json(output,{'command':list(map(str,args)),'exit_code':result.returncode,'seconds':time.perf_counter()-began,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode: raise RuntimeError(f'Command failed; see {output}')
    return result


def qualification(root,out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'environment.json',environment(root))
    atomic_json(out/'gpu_inventory.json',{'devices':[{'index':i,'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory,'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())],'training_device':'cuda:0','parallel_training':False})
    command([sys.executable,'-m','pytest','-q'],root,out/'tests.json')
    data=Path('/tmp/qualification-data')
    manifest=prepare(data,transport='http')
    reference=json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    comparison={key:manifest[key]==reference[key] for key in ['source','split','partitions','tokenizers','manifest_hash']}
    atomic_json(out/'dataset_comparison.json',{'matches':comparison,'manifest':manifest})
    assert all(comparison.values())
    verify(root,data,out/'cuda_smoke','cuda','float16')
    # Execute the documented benchmark CLI without substituting training logic.
    command([sys.executable,'-m','nano_lm.experiments','--root',str(root),'--dataset',str(data),'--benchmark-all','--output',str(out/'all14'),'--device','cuda','--steps','10'],root,out/'benchmark_command.json')
    rows=[]
    representatives=['E01','E09','E12','E14','E02']
    for eid in representatives:
        directory,checkpoint,summary=train(root,data,eid,TrainSettings(device='cuda',run_kind='benchmark'),50,out/'warmup_runs')
        payload,_=read_checkpoint(checkpoint)
        h=payload['training_state']['history'];warm=h[9];last=h[-1]
        elapsed=last['train_seconds']-warm['train_seconds'];exposure=last['training_tokens']-warm['training_tokens']
        rows.append({'experiment_id':eid,'run_id':summary['run_id'],'warmup_steps_excluded':10,'measured_steps':40,'train_seconds':elapsed,'steps_per_second':40/elapsed,'tokens_per_second':exposure/elapsed,'peak_gpu_bytes':last['peak_gpu_bytes'],'projected_5000_train_seconds':5000/(40/elapsed),'projection_limit':'training only; not measured full-run time'})
        atomic_json(out/'warmup_benchmark.json',rows) if len(rows)==1 else None
    # Publish once after the complete representative set, preserving the progress record.
    atomic_json(out/'warmup_benchmark_complete.json',rows)
    validation=[];overhead=[]
    text=(data/'validation.txt').read_text(encoding='utf-8')
    for directory in sorted((out/'all14').iterdir()):
        if not directory.is_dir():continue
        checkpoint=next((directory/'checkpoints').glob('step_*.pt'))
        model,tokenizer=load_for_inference(checkpoint);model.to('cuda')
        torch.cuda.synchronize();began=time.perf_counter()
        metric=evaluate(model,tokenizer,text)
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        validation.append({'run_id':directory.name,'experiment_id':directory.name.split('_')[0],'seconds':seconds,'metrics':metric,'label':'10-step qualification validation, not research result'})
        del model;torch.cuda.empty_cache()
        payload,_=read_checkpoint(checkpoint)
        # Timed genuine checkpoint serialization to a new artifact; no earlier file overwritten.
        copied=out/'io_measurements'/f'{directory.name}.pt';copied.parent.mkdir(exist_ok=True)
        began=time.perf_counter()
        with copied.open('xb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        serialization=time.perf_counter()-began
        began=time.perf_counter();snapshot=export_run(directory,out/'benchmark_exports');export_seconds=time.perf_counter()-began
        overhead.append({'run_id':directory.name,'checkpoint_bytes':checkpoint.stat().st_size,'serialization_fsync_seconds':serialization,'export_seconds':export_seconds,'snapshot':str(snapshot),'scope':'local /kaggle/working IO; saved-version/download overhead separately measured'})
    atomic_json(out/'full_validation.json',validation);atomic_json(out/'io_overhead.json',overhead)
    smoke=next(p for p in (out/'cuda_smoke/runs').iterdir() if p.name.startswith('E01') and (p/'checkpoints/step_000004.pt').exists())
    snapshot=export_run(smoke,out/'durable_snapshots');verify_snapshot(snapshot)
    atomic_json(out/'recovery_selection.json',{'snapshot':snapshot.name,'run_id':smoke.name,'checkpoint':'step_000004.pt','steps_before':4,'settings':{'device':'cuda','precision':'float16','run_kind':'smoke','tokens_per_update':128,'micro_batch_size':1,'eval_every':2,'checkpoint_every':1,'eval_max_bytes':512}})
    atomic_json(out/'completed.json',{'label':'qualification_not_research','status':'compute_checks_complete_storage_recovery_pending','git_commit':environment(root)['git_commit']})
    # Durable notebook outputs retain all checkpoints. A compact download bundle
    # includes complete smoke snapshot and benchmark logs/metadata, not duplicate weights.
    bundle=out.parent/'recovery_bundle.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file() and (path.suffix!='.pt' or 'durable_snapshots' in path.relative_to(out).parts):
                z.write(path,path.relative_to(out).as_posix())
    atomic_json(out.parent/'bundle_manifest.json',{'sha256':file_hash(bundle),'bytes':bundle.stat().st_size,'label':'qualification_not_research','git_commit':environment(root)['git_commit']})


def recovery(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    manifests=list(Path('/kaggle/input').rglob('recovery_selection.json'))
    if len(manifests)!=1:raise RuntimeError(f'Expected one saved qualification version, found {len(manifests)}')
    original=manifests[0].parent;selection=json.loads(manifests[0].read_text())
    snapshot=original/'durable_snapshots'/selection['snapshot'];manifest=verify_snapshot(snapshot)
    restored=restore_snapshot(snapshot,out/'restored')
    checkpoint=restored/'checkpoints'/selection['checkpoint']
    model,tokenizer=load_for_inference(checkpoint)
    ids=generate(model,torch.tensor([tokenizer.encode('To be')]),8,greedy=True)
    del model
    data=Path('/tmp/recovery-data');prepare(data,transport='http')
    settings=TrainSettings(**selection['settings'])
    resumed,after,summary=train(root,data,'E01',settings,6,out/'restored',resume=checkpoint)
    full,full_checkpoint,_=train(root,data,'E01',settings,6,out/'uninterrupted')
    a,_=read_checkpoint(after);b,_=read_checkpoint(full_checkpoint)
    exact={k:equal_nested(a[k],b[k]) for k in ['model','optimizer','scheduler','scaler','rng']}
    histories=len(a['training_state']['history'])==len(b['training_state']['history']) and all(x['training_loss']==y['training_loss'] and x.get('validation')==y.get('validation') for x,y in zip(a['training_state']['history'],b['training_state']['history']))
    assert all(exact.values()) and histories
    result={'label':'qualification_recovery_not_research','environment':environment(root),'snapshot_manifest_hash':manifest['hash'],'checkpoint_sha256':file_hash(checkpoint),'steps_before':4,'steps_after':summary['step'],'exact_state':exact,'history_equal':histories,'generated_text':tokenizer.decode(ids[0].tolist()),'mounted_saved_artifact':str(original)}
    atomic_json(out/'recovery.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['qualification','recovery'],required=True)
    a=p.parse_args()
    try:
        (qualification if a.phase=='qualification' else recovery)(a.root,a.output)
    except BaseException as e:
        if a.output.exists():
            (a.output/'failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'label':'qualification_failure_not_research'},indent=2))
        raise
