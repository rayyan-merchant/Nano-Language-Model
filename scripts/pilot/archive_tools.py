"""Shared bounded SHA256 and exact sparse-snapshot recovery for published archives."""
import hashlib
import json
import shutil
from nano_lm.storage import verify_snapshot


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def reconstruct_snapshots(extracted,prefix,output):
    reconstructed=output;reconstructed.mkdir(exist_ok=False)
    manifests=[]
    for source in (extracted/'snapshots').iterdir():
        target=reconstructed/source.name;shutil.copytree(source,target)
        manifest=json.loads((target/'export_manifest.json').read_text())
        for relative,h in manifest['files'].items():
            path=target/relative
            if not path.exists():
                assert relative.startswith('checkpoints/')
                weight=extracted/prefix/relative
                assert checksum(weight)==h
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(weight,path)
        manifests.append(verify_snapshot(target))
    return manifests
