"""E14-only persistent development qualification, using the canonical engine."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint
from nano_lm.storage import export_run, verify_snapshot, restore_snapshot
from nano_lm.train import TrainSettings, train, environment
try:
    from .preflight_restore import audit_restore
except ImportError:
    from preflight_restore import audit_restore


def require_recovery_receipt(root, out):
    paths=list(Path('/kaggle/input').rglob('development_recovery.json'))
    if len(paths)!=1:raise RuntimeError('E14 requires exactly one published fresh development CUDA recovery receipt')
    receipt=json.loads(paths[0].read_text())
    assert receipt['status']=='published_E14_development_fresh_CUDA_full_state_passed'
    assert receipt['source_commit']==environment(root)['git_commit']
    assert set(receipt['exact_state'])=={'model','optimizer','scheduler','scaler','rng','training_state'}
    assert receipt['additional_optimizer_updates']==0 and all(receipt['exact_state'].values())
    assert receipt['parameters']==2808960 and receipt['step_before']==receipt['step_after']==1
    atomic_json(out/'independent_development_recovery_gate.json',receipt)


def publish_development(root,out,data,experiment,checkpoint,env):
    assert experiment=='E14'
    settings=TrainSettings(device='cuda',run_kind='benchmark',eval_every=10,checkpoint_every=10)
    import nano_lm.storage as storage
    original_export=storage.export_run;export_times=[]
    def timed_export(*args,**kwargs):
        start=time.perf_counter();result=original_export(*args,**kwargs)
        export_times.append(time.perf_counter()-start);return result
    storage.export_run=timed_export
    began=time.perf_counter()
    try:
        _,benchmark,_=train(root,data,'E14',settings,10,out/'benchmark',export_root=out/'benchmark_snapshots')
    finally:
        storage.export_run=original_export
    payload,_=read_checkpoint(benchmark);h=payload['training_state']['history']
    assert len(h)==10 and h[-1]['accumulation']==8 and h[-1]['validation']['scored_bytes']==55514
    warm_seconds=h[-1]['train_seconds']-h[1]['train_seconds']
    projected=5000*warm_seconds/8+20*payload['training_state']['eval_seconds']+600
    hardware=json.loads((out/'hardware.json').read_text())
    assert projected<hardware['session_limit_seconds'],'Measured estimate exceeds session budget including postprocessing margin'
    atomic_json(out/'development_benchmark.json',{'label':'E14 development qualification, not research measurements',
        'optimizer_updates':10,'warmup_updates_excluded':2,'training_seconds':h[-1]['train_seconds'],
        'warmup_excluded_updates_per_second':8/warm_seconds,'warmup_excluded_tokens_per_second':8*4096/warm_seconds,
        'projected_5000_training_only_seconds':5000*warm_seconds/8,'full_validation_seconds':payload['training_state']['eval_seconds'],
        'peak_gpu_bytes':h[-1]['peak_gpu_bytes'],'checkpoint_bytes':benchmark.stat().st_size,
        'benchmark_wall_seconds':time.perf_counter()-began,'snapshot_export_seconds':export_times,
        'projected_training_validation_plus_600s_margin':projected,
        'scope':'Short benchmark projection is not measured full-run time'})
    began=time.perf_counter()
    snapshot=export_run(checkpoint.parent.parent,out/'development_snapshots')
    manifest=verify_snapshot(snapshot)
    snapshot_bytes=sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file())
    seconds=time.perf_counter()-began
    atomic_json(out/'development_export_timing.json',{'snapshot_bytes':snapshot_bytes,'export_and_verify_seconds':seconds,
        'bytes_per_second':snapshot_bytes/seconds,'scope':'Actual local immutable export and hash verification; publication/download measured separately'})
    atomic_json(out/'development_completion.json',{'experiment':'E14','run_kind':'smoke','step':1,
        'source_commit':env['git_commit'],'code_hash':env['code_hash'],'run_id':checkpoint.parent.parent.name,
        'checkpoint':str(checkpoint.relative_to(out)),'snapshot':str(snapshot.relative_to(out)),
        'snapshot_hash':manifest['hash'],'research_updates':0})
    archive=out.parent/'E14_development.zip'
    began=time.perf_counter()
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in files:z.write(out/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(files)
    atomic_json(out.parent/'artifact_manifest.json',{'archive_file':archive.name,'source_commit':env['git_commit'],
        'run_id':checkpoint.parent.parent.name,'run_kind':'development','archive_sha256':file_hash(archive),
        'archive_bytes':archive.stat().st_size,'files':files,'research_updates':0,'archive_and_CRC_seconds':time.perf_counter()-began})
    print('E14 DEVELOPMENT PUBLICATION READY',flush=True)


def recover_development(root,out,original=None):
    out.mkdir(parents=True,exist_ok=False)
    if original is None:
        paths=list(Path('/kaggle/input').rglob('development_completion.json'))
        assert len(paths)==1;original=paths[0].parent
    manifest_path=original.parent/'artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text());completion=json.loads((original/'development_completion.json').read_text())
    env=environment(root)
    assert completion['experiment']=='E14' and completion['research_updates']==0
    assert completion['source_commit']==env['git_commit'] and not env['git_dirty']
    assert env['python']=='3.11.14' and env['torch']=='2.5.1+cu124' and env['cuda']=='12.4' and env['gpu']=='Tesla T4'
    assert env['code_hash']==completion['code_hash']=='cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'
    normalize=lambda lines:sorted(x for x in lines if x and not x.startswith('-e '))
    assert normalize(env['packages'])==normalize((root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines())
    archive=original.parent/manifest['archive_file']
    assert file_hash(archive)==manifest['archive_sha256'] and archive.stat().st_size==manifest['archive_bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest['files'])
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
    for name,sha in manifest['files'].items():assert file_hash(original/name)==sha
    snapshot=original/completion['snapshot'];verified=verify_snapshot(snapshot)
    assert verified['hash']==completion['snapshot_hash']
    restored=restore_snapshot(snapshot,out/'restored')
    checkpoint=restored/'checkpoints'/Path(completion['checkpoint']).name
    receipt=audit_restore(checkpoint,out/'state_roundtrip.pt','cuda')
    receipt.update({'status':'published_E14_development_fresh_CUDA_full_state_passed','source_commit':env['git_commit'],
        'environment':env,'archive_sha256':manifest['archive_sha256'],'verified_file_count':len(manifest['files']),
        'checkpoint_sha256':file_hash(checkpoint),'resolved':read_checkpoint(checkpoint)[0]['metadata']['resolved'],
        'snapshot_hash':verified['hash'],'original_kernel':json.loads((original/'hardware.json').read_text())['kernel_id']})
    atomic_json(out/'development_recovery.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();recover_development(a.root,a.output)
