"""Scoped E07 Colab orchestration using the unchanged canonical engine."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import torch
from nano_lm.artifacts import atomic_json, file_hash, digest
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.prepare import prepare, validate
from nano_lm.storage import snapshot_file, verify_snapshot, restore_snapshot
from nano_lm.train import TrainSettings, environment, train
from e07_heads_audit import audit

CORE = 'cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'
SOURCE = '5bb03b18943e7c45a305ce1d78b3446c31829ae7'


def publish_json(path, value):
    snapshot_file(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def qualified(root):
    env = environment(root)
    assert not env['git_dirty'] and env['git_commit'] == SOURCE and env['code_hash'] == CORE
    assert env['python'] == '3.11.14' and env['torch'] == '2.5.1+cu124' and env['cuda'] == '12.4'
    assert torch.cuda.is_available() and env['gpu'] == 'Tesla T4'
    expected = (root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines()
    norm = lambda lines: sorted(x for x in lines if x and not x.startswith('-e '))
    assert norm(env['packages']) == norm(expected)
    assert len(norm(env['packages'])) == 46
    return env


def preflight(root, out, store, quota_free):
    started = time.perf_counter(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); env = qualified(root)
    atomic_json(out/'environment.json', env)
    hardware = {'gpu_count': torch.cuda.device_count(), 'training_device': 'cuda:0',
                'devices': [{'name':torch.cuda.get_device_name(i), 'total_bytes':torch.cuda.get_device_properties(i).total_memory,
                            'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())],
                'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
                'local_disk_free_bytes':shutil.disk_usage(out).free}
    assert hardware['devices'][0]['free_bytes'] > 2*2**30 and hardware['local_disk_free_bytes'] > 5*2**30
    atomic_json(out/'hardware.json', hardware)
    atomic_json(out/'heads_audit.json', audit(root, 'cuda'))
    result = subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,capture_output=True,text=True)
    atomic_json(out/'tests.json', {'returncode':result.returncode,'output':result.stdout+result.stderr})
    result.check_returncode()
    # Run the additional attention test against the clean trained source from an external script directory.
    test = subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve().parents[2]/'tests/test_e07_heads_research.py')],cwd=root,capture_output=True,text=True)
    atomic_json(out/'attention_test.json', {'returncode':test.returncode,'output':test.stdout+test.stderr})
    test.check_returncode()
    data = out/'data'; manifest = prepare(data); validate(data)
    reference = json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    assert manifest == reference
    assert manifest['source']['sha256'] == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    assert manifest['tokenizers']['character']['hash'] == 'ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77'
    assert [manifest['partitions'][p]['bytes'] for p in ['train','validation','test']] == [1003854,55770,55770]
    atomic_json(out/'dataset_verification.json', {'all_reference_fields_match':True,'manifest':manifest})
    assert store.is_dir() and str(store).startswith('/content/drive/MyDrive/Nano-Language-Model/E07_seed42/')
    publish_json(store/'write_probe.json', {'experiment':'E07','source_commit':SOURCE})
    assert json.loads((store/'write_probe.json').read_text())['experiment'] == 'E07'
    # Two labeled smoke updates, full architecture/effective batch, capped validation.
    settings = TrainSettings(seed=42,device='cuda',run_kind='smoke',eval_every=1,checkpoint_every=1,eval_max_bytes=512)
    run, checkpoint, _ = train(root,data,'E07',settings,2,out/'smoke_runs',export_root=store/'smoke_snapshots')
    snapshots = [(p,verify_snapshot(p)) for p in (store/'smoke_snapshots').iterdir()]
    latest, m = max(snapshots,key=lambda pm:(pm[1]['step'],len(pm[1]['files'])))
    restored = restore_snapshot(latest,out/'smoke_restore')
    restored_checkpoint = restored/'checkpoints/step_000002.pt'
    assert file_hash(restored_checkpoint) == file_hash(checkpoint)
    model,tok = load_for_inference(restored_checkpoint); model.to('cuda')
    text = tok.decode(generate(model,torch.tensor([tok.encode('To be')],device='cuda'),8,greedy=True)[0].tolist())
    del model; torch.cuda.empty_cache()
    # Verify complete-state continuation on the separate smoke lineage only.
    _, continued, _ = train(root,data,'E07',settings,4,out/'unused_smoke_output',resume=restored_checkpoint,export_root=store/'smoke_resume_snapshots')
    assert read_checkpoint(continued)[0]['step'] == 4
    conservative = checkpoint.stat().st_size*100 + 1024**3
    assert quota_free > conservative and hardware['local_disk_free_bytes'] > conservative
    record = {'status':'passed','experiment':'E07','seed':42,'authorized_research_updates':5000,
              'source_commit':SOURCE,'code_hash':CORE,'parameters':1804800,'dependency_packages':46,
              'actual_drive_quota_free_bytes':quota_free,'quota_source':'Google Drive storage UI observed before launch',
              'smoke_checkpoint_bytes':checkpoint.stat().st_size,'conservative_artifact_budget_bytes':conservative,
              'smoke_verified_snapshot_count':len(snapshots),'fresh_restore_hash_equal':True,
              'fresh_cuda_smoke_inference':text,'restored_smoke_total_step':4,'research_updates':0,
              'storage':str(store),'seconds':time.perf_counter()-started}
    atomic_json(out/'preflight.json',record)
    for p in out.glob('*.json'): snapshot_file(store/p.name,p.read_bytes())
    print('E07 PREFLIGHT PASSED',json.dumps(record),flush=True)


def research(root,out,store,resume=None):
    qualified(root); assert json.loads((out/'preflight.json').read_text())['status']=='passed'
    if resume is None:
        assert not (out/'runs').exists() and not (store/'research_snapshots').exists()
    else:
        gate=json.loads((out/'resume_preflight.json').read_text())
        assert gate['status']=='hash_verified_full_state_CUDA_restoration_passed'
        assert file_hash(resume)==gate['checkpoint_sha256']
        assert read_checkpoint(resume)[0]['step']==gate['checkpoint_step']<5000
    import nano_lm.train as engine
    import nano_lm.storage as storage
    timers=[]; journal=[0.]
    def timed(name, function):
        def call(*args,**kwargs):
            began=time.perf_counter(); result=function(*args,**kwargs)
            timers.append({'operation':name,'seconds':time.perf_counter()-began,'reference':str(result if name=='export' else args[0])})
            return result
        return call
    engine.save_checkpoint=timed('checkpoint',engine.save_checkpoint)
    storage.export_run=timed('export',storage.export_run)
    original=engine.append_event
    def event(path,value):
        began=time.perf_counter(); original(path,value); journal[0]+=time.perf_counter()-began
        if value['event']=='step' and value['metric']['step']%250==0:
            print('E07 PROGRESS',json.dumps(value['metric']),flush=True)
    engine.append_event=event
    settings=TrainSettings(seed=42,device='cuda',precision='float32',run_kind='research',eval_every=250,checkpoint_every=500)
    atomic_json(out/'execution.json',{'entrypoint':'nano_lm.train.train','settings':asdict(settings),'steps':5000,
                                    'source_commit':SOURCE,'resume_checkpoint':str(resume) if resume else None,'timing_hooks':'Forward unchanged calls; no numerical or RNG changes'})
    began=time.perf_counter()
    run,checkpoint,summary=engine.train(root,out/'data','E07',settings,5000,out/'runs',resume=resume,export_root=store/'research_snapshots')
    atomic_json(out/'io_timing.json',{'records':timers,'journal_seconds':journal[0],'engine_wall_seconds':time.perf_counter()-began,'scope':'resumed segment only; prior checkpoint/export overhead unavailable' if resume else 'complete original research segment'})
    atomic_json(out/'completion.json',{'status':'5000_completed_optimizer_updates','run_id':run.name,
                                      'final_checkpoint':str(checkpoint.relative_to(out)),'best_checkpoint':summary['best_checkpoint'],
                                      'source_commit':SOURCE,'code_hash':CORE,'validation_steps':list(range(250,5001,250))})
    for name in ['execution.json','io_timing.json','completion.json']:
        snapshot_file(store/name,(out/name).read_bytes())
    print('E07 EXACTLY 5000 UPDATES COMPLETE',flush=True)


def finish(root,out,store):
    began=time.perf_counter(); qualified(root)
    snapshots=[(p,verify_snapshot(p)) for p in (store/'research_snapshots').iterdir()]
    assert sorted(set(m['step'] for _,m in snapshots))==list(range(250,5001,250))
    latest,m=max(snapshots,key=lambda pm:(pm[1]['step'],len(pm[1]['files'])))
    run=restore_snapshot(latest,out/'independent_runs')
    for p,manifest in snapshots:
        for relative,checksum in manifest['files'].items():
            if relative.startswith('checkpoints/'):
                target=run/relative
                if not target.exists():shutil.copyfile(p/relative,target)
                assert file_hash(target)==checksum
    final,tok=read_checkpoint(run/'checkpoints/step_005000.pt'); state=final['training_state'];h=state['history']
    assert final['step']==5000 and final['metadata']['parameters']==1804800
    assert final['metadata']['resolved']['experiment_id']=='E07' and final['config']['activation']=='gelu' and final['config']['n_heads']==8
    assert [r['step'] for r in h]==list(range(1,5001))
    val=[r for r in h if 'validation' in r]
    assert [r['step'] for r in val]==list(range(250,5001,250))
    assert state['training_tokens']==state['training_bytes']==20480000 and state['amp_skipped_updates']==0
    assert all(r['lr']==.0003 and r['accumulation']==8 and math.isfinite(r['gradient_norm']) for r in h)
    for row in val:
        v=row['validation']; assert v['scored_bytes']==v['scored_tokens']==55514 and v['conditioning_bytes']==256
        assert v['evaluated_text_bytes']==55770 and v['mode']=='sliding' and v['context_length']==128
        assert math.isclose(v['bpb'],v['loss']/math.log(2),rel_tol=1e-12)
    checks=[]
    for p in sorted((run/'checkpoints').glob('step_*.pt')):
        payload,_=read_checkpoint(p);step=payload['step']
        counters=sorted(set(int(v['step'].item()) for v in payload['optimizer']['state'].values()))
        assert counters==[step] and payload['scheduler']['last_epoch']==step
        assert payload['metadata']==final['metadata'] and len(payload['training_state']['history'])==step
        checks.append({'file':p.name,'step':step,'sha256':file_hash(p),'optimizer_counters':counters,'scheduler_epoch':step})
    assert len(checks)==20
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='step' for e in events)==5000
    atomic_json(out/'checkpoint_audit.json',{'status':'passed','run_id':run.name,'checkpoint_count':20,
                 'snapshot_count':len(snapshots),'checks':checks,'validation_steps':[r['step'] for r in val],
                 'journal_failures_or_resumes':[e for e in events if e['event'] in ['failure','resume']],
                 'snapshot_restore_counter_seconds':time.perf_counter()-began})
    atomic_json(out/'snapshot_index.json',[{'directory':p.name,**manifest} for p,manifest in snapshots])
    # Reuse the reviewed complete CUDA optimizer/RNG roundtrip audit in a fresh subprocess.
    (out/'snapshots').symlink_to(store/'research_snapshots',target_is_directory=True)
    code="from pathlib import Path; from recover import recover; recover(Path("+repr(str(root))+"),Path("+repr(str(out/'cuda_recovery'))+"),Path("+repr(str(out))+"))"
    subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(root/'scripts/pilot')},cwd=root,check=True)
    subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(run),'--output',str(out/'analysis'),'--device','cuda'],cwd=root,check=True)
    norms=np.array([r['gradient_norm'] for r in h])
    atomic_json(out/'gradient_diagnostics.json',{'finite_steps':len(h),'norm_mean':float(norms.mean()),'norm_median':float(np.median(norms)),
                 'norm_min':float(norms.min()),'norm_max':float(norms.max()),'steps_above_clipping_threshold':int((norms>1).sum()),
                 'clipping_frequency':float((norms>1).mean()),'final_norm':float(norms[-1])})
    # Publish analysis and small evidence first. Never use hard links on Drive.
    for p in (out/'analysis').rglob('*'):
        if p.is_file():snapshot_file(store/'analysis'/p.relative_to(out/'analysis'),p.read_bytes())
    for p in out.glob('*.json'):
        if not (store/p.name).exists():snapshot_file(store/p.name,p.read_bytes())
    snapshot_file(store/'cuda_recovery.json',(out/'cuda_recovery/recovery.json').read_bytes())
    atomic_json(out/'postprocessing_wall_time.json', {'seconds_before_archive':time.perf_counter()-began, 'scope':'Verified snapshot restore/counter audit, fresh CUDA optimizer/RNG recovery, complete GPU generation, attention evidence preservation and Kaggle local artifact staging; excludes training, bootstrap, Drive upload and independent retrieval'})
    package(root,out,store,run)
    publish_json(store/'postprocessing_complete.json', {'finish_phase_seconds':time.perf_counter()-began, 'additional_optimizer_updates':0, 'scope':'Complete finish phase including verification, analysis, generation, archive construction/publication and numbered parts; excludes bootstrap, training and independent retrieval'})
    print('E07 RECOVERY GENERATION AND PUBLICATION PASSED',flush=True)


def package(root,out,store,run):
    started=time.perf_counter();archive=out.parent/'E07_seed42_complete.zip'
    included=list(run.rglob('*'))+list((out/'analysis').rglob('*'))+list(out.glob('*.json'))+[out/'cuda_recovery/recovery.json']
    files=[p for p in included if p.is_file()]
    # Canonical archive paths keep completion.json directly usable after extraction.
    sources={('runs/'+run.name+'/'+p.relative_to(run).as_posix() if p.is_relative_to(run) else p.relative_to(out).as_posix()):p for p in files}
    assert len(sources)==len(files)
    # Preserve every snapshot's exact small metadata/journal/manifest, sharing weight bytes.
    for snapshot in (store/'research_snapshots').iterdir():
        for path in snapshot.rglob('*'):
            if path.is_file() and path.suffix!='.pt':
                name='snapshots/'+snapshot.name+'/'+path.relative_to(snapshot).as_posix()
                assert name not in sources
                sources[name]=path
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in sources.items():z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        hashes={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
    assert all(file_hash(sources[name])==h for name,h in hashes.items())
    snapshot_file(store/archive.name,archive.read_bytes())
    assert file_hash(store/archive.name)==file_hash(archive)
    manifest={'run_id':run.name,'run_prefix':'runs/'+run.name+'/', 'source_commit':SOURCE,
              'archive_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),'files':hashes,
              'crc_and_source_hashes_verified':True,'archive_and_publication_seconds':time.perf_counter()-started}
    atomic_json(out/'artifact_manifest.json',manifest);publish_json(store/'artifact_manifest.json',manifest)
    parts=[];partdir=store/'archive_parts';partdir.mkdir(exist_ok=False)
    with archive.open('rb') as f:
        for i,chunk in enumerate(iter(lambda:f.read(200*2**20),b'')):
            name=f'part_{i:02d}.bin';snapshot_file(partdir/name,chunk)
            checksum=hashlib.sha256(chunk).hexdigest();assert file_hash(partdir/name)==checksum
            parts.append({'name':name,'bytes':len(chunk),'sha256':checksum})
    receipt={'archive':archive.name,'archive_bytes':manifest['archive_bytes'],'archive_sha256':manifest['archive_sha256'],'parts':parts}
    atomic_json(out/'archive_parts.json',receipt);publish_json(store/'archive_parts.json',receipt)
    print('E07 ARCHIVE',json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['preflight','research','finish'])
    for name in ['root','output','store']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--quota-free-bytes',type=int)
    a=p.parse_args()
    try:
        if a.phase=='preflight':preflight(a.root,a.output,a.store,a.quota_free_bytes)
        elif a.phase=='research':research(a.root,a.output,a.store)
        else:finish(a.root,a.output,a.store)
    except BaseException as exc:
        if a.output.exists():
            failure={'phase':a.phase,'type':type(exc).__name__,'message':str(exc)}
            (a.output/(a.phase+'_failure.json')).write_text(json.dumps(failure,indent=2))
        raise
