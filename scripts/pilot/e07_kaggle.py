"""E07 platform qualification and execution; canonical scientific engine unchanged."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint, load_for_inference, restore_training, save_checkpoint
from nano_lm.config import ModelConfig
from nano_lm.generate import generate
from nano_lm.model import GPT
from nano_lm.prepare import prepare, validate
from nano_lm.storage import export_run, verify_snapshot, restore_snapshot
from nano_lm.train import TrainSettings, train
from nano_lm.verification import equal_nested
from e07_colab import qualified, research, finish, CORE, SOURCE
from e07_heads_audit import audit


def state_restore(checkpoint, output):
    payload,tok=read_checkpoint(checkpoint)
    model=GPT(ModelConfig(**payload['config'])).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=False);generator=torch.Generator()
    restore_training(payload,model,optimizer,scheduler,generator);scaler.load_state_dict(payload['scaler'] or {})
    save_checkpoint(output,model,tok,optimizer,scheduler,generator,payload['step'],payload['metadata'],scaler,payload['training_state'])
    actual,_=read_checkpoint(output)
    exact={k:equal_nested(actual[k],payload[k]) for k in ['model','optimizer','scheduler','scaler','rng','training_state']}
    assert all(exact.values())
    model.eval();ids=generate(model,torch.tensor([tok.encode('ROMEO:\n')],device='cuda'),16,greedy=True)
    return {'status':'passed','checkpoint_sha256':file_hash(checkpoint),'step':payload['step'],'exact_state':exact,'text':tok.decode(ids[0].tolist()),'additional_optimizer_updates':0}


def preflight(root,out):
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    env=qualified(root);atomic_json(out/'environment.json',env)
    hardware={'gpu_count':torch.cuda.device_count(),'training_device':'cuda:0','nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'local_disk_free_bytes':shutil.disk_usage(out).free,'devices':[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory,'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())]}
    assert hardware['local_disk_free_bytes']>5*2**30 and hardware['devices'][0]['free_bytes']>2*2**30
    atomic_json(out/'hardware.json',hardware);atomic_json(out/'heads_audit.json',audit(root,'cuda'))
    for name,args in [('tests',[sys.executable,'-m','pytest','-q']),('attention_test',[sys.executable,'-m','pytest','-q',str(Path(__file__).resolve().parents[2]/'tests/test_e07_heads_research.py')])]:
        result=subprocess.run(args,cwd=root,capture_output=True,text=True)
        atomic_json(out/(name+'.json'),{'returncode':result.returncode,'output':result.stdout+result.stderr});result.check_returncode()
    manifest=prepare(out/'data');validate(out/'data')
    assert manifest==json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    atomic_json(out/'dataset_verification.json',{'all_reference_fields_match':True,'manifest':manifest})
    settings=TrainSettings(seed=42,device='cuda',run_kind='smoke',eval_every=1,checkpoint_every=1,eval_max_bytes=512)
    _,checkpoint,_=train(root,out/'data','E07',settings,2,out/'smoke_runs',export_root=out/'smoke_snapshots')
    snapshots=[(p,verify_snapshot(p)) for p in (out/'smoke_snapshots').iterdir()]
    latest,m=max(snapshots,key=lambda pm:(pm[1]['step'],len(pm[1]['files'])))
    restored=restore_snapshot(latest,out/'smoke_restored')
    cp=restored/'checkpoints/step_000002.pt';assert file_hash(cp)==file_hash(checkpoint)
    receipt=state_restore(cp,out/'roundtrip_step_000002.pt');atomic_json(out/'local_smoke_recovery.json',receipt)
    _,continued,_=train(root,out/'data','E07',settings,4,out/'unused_smoke_root',resume=cp,export_root=out/'smoke_resume_snapshots')
    assert read_checkpoint(continued)[0]['step']==4
    budget=checkpoint.stat().st_size*100+2**30;assert hardware['local_disk_free_bytes']>budget
    record={'status':'local_preflight_passed_persistent_publication_pending','experiment':'E07','seed':42,'source_commit':SOURCE,'code_hash':CORE,'parameters':1804800,'dependency_packages':46,'smoke_checkpoint_bytes':checkpoint.stat().st_size,'conservative_artifact_budget_bytes':budget,'local_restored_state_equal':True,'smoke_verified_snapshot_count':len(snapshots),'smoke_total_optimizer_step':4,'research_updates':0,'seconds':time.perf_counter()-began,'storage_policy':'Kaggle saved output version must be independently retrieved and mounted/recovered in a fresh compatible GPU process before research. /kaggle/working alone is ephemeral.'}
    atomic_json(out/'preflight.json',record)
    # Compact source snapshot is immutable step2, independently restorable without data.
    archive=out.parent/'E07_smoke_bundle.zip';files={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in latest.rglob('*'):
            if p.is_file():
                name='snapshot/'+p.relative_to(latest).as_posix();z.write(p,name);files[name]=file_hash(p)
        for p in out.glob('*.json'):
            z.write(p,p.name);files[p.name]=file_hash(p)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    atomic_json(out.parent/'E07_smoke_manifest.json',{'archive_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),'files':files,'source_commit':SOURCE,'step':2,'scope':'Labeled smoke only; no research results'})
    print('E07 LOCAL PREFLIGHT PASSED; SAVE OUTPUT AND INDEPENDENTLY RECOVER BEFORE RESEARCH',json.dumps(record),flush=True)


def recover_smoke(root,out,archive,manifest):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);env=qualified(root)
    receipt=json.loads(manifest.read_text());assert archive.stat().st_size==receipt['archive_bytes'] and file_hash(archive)==receipt['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(receipt['files'])
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        for name,h in receipt['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        z.extractall(out/'published')
    snapshot=out/'published/snapshot';verify_snapshot(snapshot);restored=restore_snapshot(snapshot,out/'fresh_restore')
    cp=restored/'checkpoints/step_000002.pt'
    actual=state_restore(cp,out/'roundtrip_step_000002.pt')
    atomic_json(out/'persistent_recovery.json',{'status':'saved_output_archive_and_fresh_compatible_CUDA_full_state_recovery_passed','environment':env,'archive_sha256':receipt['archive_sha256'],'archive_bytes':receipt['archive_bytes'],'recovery':actual,'additional_research_optimizer_updates':0,'scope':'Published Kaggle output mounted as read-only input; fresh isolated compatible process restored complete smoke state.'})
    print('E07 PERSISTENT CUDA SMOKE RECOVERY PASSED',json.dumps(actual),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','recover-smoke','research','finish']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--store',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--recovery-receipt',type=Path)
    a=p.parse_args()
    try:
        if a.phase=='preflight':preflight(a.root,a.output)
        elif a.phase=='recover-smoke':recover_smoke(a.root,a.output,a.archive,a.manifest)
        elif a.phase=='research':
            gate=json.loads(a.recovery_receipt.read_text());assert gate['status']=='saved_output_archive_and_fresh_compatible_CUDA_full_state_recovery_passed'
            a.output.mkdir(parents=True,exist_ok=False)
            qualified(a.root)
            # Import verified saved smoke evidence; reacquire canonical data in this runtime.
            for path in (a.recovery_receipt.parent/'published').glob('*.json'):
                if path.name not in {'preflight.json','environment.json','hardware.json'}:shutil.copyfile(path,a.output/path.name)
            current=qualified(a.root);atomic_json(a.output/'environment.json',current)
            hardware={'gpu_count':torch.cuda.device_count(),'training_device':'cuda:0','nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'local_disk_free_bytes':shutil.disk_usage(a.output).free,'devices':[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory,'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())]}
            assert hardware['local_disk_free_bytes']>5*2**30 and hardware['devices'][0]['free_bytes']>2*2**30
            atomic_json(a.output/'hardware.json',hardware)
            manifest=prepare(a.output/'data');validate(a.output/'data')
            assert manifest==json.loads((a.root/'docs/evidence/dataset_manifest.json').read_text())
            atomic_json(a.output/'preflight.json',{'status':'passed','platform':'Kaggle','persistent_recovery':gate,'scope':'Saved smoke version independently retrieved before launch and recovered in this fresh compatible CUDA process. Research fresh initialization; no weights transferred from smoke.'})
            a.store.mkdir(parents=True,exist_ok=False)
            research(a.root,a.output,a.store)
        else:finish(a.root,a.output,a.store)
    except BaseException as exc:
        if a.output.exists():
            (a.output/(a.phase+'_failure.json')).write_text(json.dumps({'phase':a.phase,'type':type(exc).__name__,'message':str(exc)},indent=2))
        raise
