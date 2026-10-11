"""Verify snapshot metadata and reconstruct omitted identical weight copies."""
import json
import os
from pathlib import Path
from nano_lm.artifacts import file_hash
from nano_lm.storage import verify_snapshot


def complete_snapshots(root):
    root=Path(root);runs=list((root/'runs').iterdir())
    if len(runs)!=1:raise ValueError('Expected exactly one preserved run')
    run=runs[0];records=[]
    for snapshot in (root/'snapshots').iterdir():
        manifest=json.loads((snapshot/'export_manifest.json').read_text())
        for relative,checksum in manifest['files'].items():
            target=snapshot/relative
            if snapshot.resolve() not in target.resolve().parents:raise ValueError('Unsafe snapshot path')
            if not target.exists():
                source=run/relative
                if not relative.startswith('checkpoints/') or (run/'checkpoints').resolve() not in source.resolve().parents:
                    raise ValueError('Only duplicate checkpoint copies may be reconstructed')
                if file_hash(source)!=checksum:raise ValueError('Source checkpoint checksum mismatch')
                target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
        verified=verify_snapshot(snapshot);records.append(verified)
    return records
