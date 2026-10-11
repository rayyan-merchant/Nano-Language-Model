"""Publishable compact rescue evidence from downloaded original notebook outputs."""
import argparse
import json
import os
from pathlib import Path
import uuid
import zipfile
from nano_lm.artifacts import file_hash, atomic_json
from nano_lm.checkpoint import read_checkpoint
from nano_lm.storage import verify_snapshot


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False);runs=list((source/'runs').iterdir());assert len(runs)==1
    run=runs[0];checkpoints=list((run/'checkpoints').glob('step_*.pt'));assert len(checkpoints)==20
    for checkpoint in checkpoints:
        p,_=read_checkpoint(checkpoint);assert p['step']==int(checkpoint.stem.split('_')[1])
        assert len(p['training_state']['history'])==p['step']
    # Rehydrate exact original snapshots from identical checkpoint bytes.
    for snapshot in (source/'snapshots').iterdir():
        m=json.loads((snapshot/'export_manifest.json').read_text())
        for name,checksum in m['files'].items():
            target=snapshot/name
            if not target.exists():
                assert name.startswith('checkpoints/') and file_hash(run/name)==checksum
                target.parent.mkdir(parents=True,exist_ok=True);os.link(run/name,target)
        verify_snapshot(snapshot)
    archive=out/'original_pilot.zip'
    files=[p for p in source.rglob('*') if p.is_file() and not ('snapshots' in p.relative_to(source).parts and p.suffix=='.pt')]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in files:z.write(path,path.relative_to(source).as_posix())
    atomic_json(out/'original_artifact_manifest.json',{'run_id':run.name,'files':{p.relative_to(source).as_posix():file_hash(p) for p in files},'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,'policy':'All20 checkpoints once plus original snapshot metadata; exact snapshot weight copies rehydrated and hash-verified from the corresponding run checkpoints'})
    identifier='rayyannmerchant/nano-e01-pilot-rescue-20261012-'+uuid.uuid4().hex[:8]
    atomic_json(out/'dataset-metadata.json',{'title':'Private E01 seed42 pilot recovery evidence','id':identifier,'licenses':[{'name':'other'}],'description':'Private engineering preservation of the genuine 5000-update E01 seed42 pilot. Original plotting failure preserved; no retraining or additional optimizer updates. Archive SHA256 and every file hash verified. Source notebook nano-e01-pilot-20261012-37785014 version1.'})
    print(json.dumps({'dataset':identifier,'archive_bytes':archive.stat().st_size,'sha256':file_hash(archive)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();prepare(a.source,a.output)
