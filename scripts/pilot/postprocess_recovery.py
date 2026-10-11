"""Recover saved 5000-update evidence and finish plotting/packaging; never train."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.storage import verify_snapshot
from nano_lm.train import environment
from recover import recover
from rehydrate import complete_snapshots


def postprocess(root,out):
    began=time.perf_counter()
    sources=list(Path('/kaggle/input').rglob('completion.json'))
    if sources:
        assert len(sources)==1;original=sources[0].parent
        records=list(Path('/kaggle/input').rglob('original_artifact_manifest.json'))
        if records:
            assert len(records)==1;record=json.loads(records[0].read_text())
            for name,checksum in record['files'].items():assert file_hash(original/name)==checksum
    else:
        archives=list(Path('/kaggle/input').rglob('original_pilot.zip'));assert len(archives)==1
        archive=archives[0];record=json.loads((archive.parent/'original_artifact_manifest.json').read_text())
        assert file_hash(archive)==record['archive_sha256'] and archive.stat().st_size==record['archive_bytes']
        original=Path('/tmp/recovered-pilot-original');original.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and set(z.namelist())==set(record['files'])
            for name in z.namelist():assert original.resolve() in (original/name).resolve().parents
            z.extractall(original)
        for name,checksum in record['files'].items():assert file_hash(original/name)==checksum
        run=next((original/'runs').iterdir())
        for snapshot in (original/'snapshots').iterdir():
            m=json.loads((snapshot/'export_manifest.json').read_text())
            for relative,checksum in m['files'].items():
                target=snapshot/relative
                if not target.exists():
                    assert relative.startswith('checkpoints/')
                    source=run/relative;assert file_hash(source)==checksum
                    target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
            verify_snapshot(snapshot)
    # Preserve original failure and all source artifacts; no train() call here.
    shutil.copytree(original,out)
    # Kaggle may automatically expand uploaded ZIPs. Always rebuild duplicate
    # weight copies in writable output, never mutate mounted read-only inputs.
    complete_snapshots(out)
    assert json.loads((out/'completion.json').read_text())['status']=='5000_completed_optimizer_updates'
    if (out/'analysis').exists():(out/'analysis').rename(out/'retained_partial_analysis')
    env=environment(root);assert not env['git_dirty']
    original_env=json.loads((out/'environment.json').read_text())
    assert env['code_hash']==original_env['code_hash'] and env['gpu']==original_env['gpu']
    normalize=lambda lines:sorted(x for x in lines if not x.startswith('-e '))
    assert normalize(env['packages'])==normalize(original_env['packages'])
    completion=json.loads((out/'completion.json').read_text());run=out/completion['final_checkpoint'];run=run.parent.parent
    analyses=list(Path('/kaggle/input').rglob('analysis_artifact_manifest.json'))
    if analyses:
        assert len(analyses)==1;receipt=json.loads(analyses[0].read_text());source=analyses[0].parent/'analysis'
        if not source.is_dir():
            packed=analyses[0].parent/'analysis.zip'
            assert file_hash(packed)==receipt['archive_sha256'] and packed.stat().st_size==receipt['archive_bytes']
            source=Path('/tmp/recovered-analysis');source.mkdir(exist_ok=False)
            with zipfile.ZipFile(packed) as z:
                assert z.testzip() is None and set(z.namelist())==set(receipt['files'])
                for name in z.namelist():assert source.resolve() in (source/name).resolve().parents
                z.extractall(source)
        for name,checksum in receipt['files'].items():assert file_hash(source/name)==checksum
        summary=json.loads((source/'summary.json').read_text());assert summary['run_id']==completion['run_id']
        assert summary['source_commit']==original_env['git_commit'] and summary['code_hash']==env['code_hash']
        generation=json.loads((source/'generation.json').read_text())
        assert generation['protocol']==json.loads((root/'configs/generation_protocol.json').read_text()) and len(generation['samples'])==100
        for sample in generation['samples']:
            assert file_hash(run/'checkpoints'/sample['checkpoint'])==sample['checkpoint_sha256']
        shutil.copytree(source,out/'analysis')
    else:
        subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(run),'--output',str(out/'analysis'),'--device','cuda'],check=True)
    recover(root,out/'recovery_audit',out)
    snapshots=[]
    for path in (out/'snapshots').iterdir():
        m=verify_snapshot(path);snapshots.append({'snapshot':path.name,'step':m['step'],'hash':m['hash'],'files':m['files']})
    atomic_json(out/'snapshot_index.json',snapshots)
    atomic_json(out/'postprocessing_recovery.json',{'status':'saved5000step_output_recovered_analyzed_and_packaged','additional_optimizer_updates':0,'source_environment':env,'original_kernel_id':json.loads((out/'hardware.json').read_text())['kernel_id'],'recovery_kernel_id':os.environ['NANO_KERNEL_ID'],'original_failure_preserved':'failure.json','backend_fix':'MPLBACKEND=Agg before importing Matplotlib; no dependency/training policy change','seconds_before_archive':time.perf_counter()-began})
    archive=out.parent/'E01_seed42_pilot.zip';started=time.perf_counter()
    def include(path):
        parts=path.relative_to(out).parts
        return path.is_file() and 'snapshots' not in parts and 'preflight_smoke' not in parts and 'recovery_audit' not in parts
    files=[p for p in out.rglob('*') if include(p)]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in files:z.write(path,path.relative_to(out).as_posix())
    checksums={p.relative_to(out).as_posix():file_hash(p) for p in files}
    atomic_json(out.parent/'artifact_manifest.json',{'source_commit':original_env['git_commit'],'postprocessing_commit':env['git_commit'],'run_id':completion['run_id'],'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,'files':checksums,'archive_seconds':time.perf_counter()-started,'snapshot_policy':'all originals and immutable snapshots preserved in both saved outputs; archive contains complete run/all20checkpoints once, analyses, original failure and snapshot index'})
    print('PILOT RECOVERY COMPLETE ARCHIVE',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();postprocess(a.root,a.output)
