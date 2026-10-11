"""Verify a downloaded saved-version bundle before inspecting or loading it."""
import argparse
import json
from pathlib import Path
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import load_for_inference
from nano_lm.generate import generate
from nano_lm.storage import verify_snapshot, restore_snapshot

def inspect(archive,manifest,output):
    expected=json.loads(manifest.read_text())
    assert file_hash(archive)==expected['sha256'] and archive.stat().st_size==expected['bytes']
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not (output/name).resolve().is_relative_to(output.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(output)
    snapshots=list((output/'durable_snapshots').iterdir())
    verified=[verify_snapshot(p) for p in snapshots]
    selection=json.loads((output/'recovery_selection.json').read_text())
    restored=restore_snapshot(output/'durable_snapshots'/selection['snapshot'],output/'local_restore')
    checkpoint=restored/'checkpoints'/selection['checkpoint']
    torch.set_num_threads(1)
    model,tokenizer=load_for_inference(checkpoint)
    ids=generate(model,torch.tensor([tokenizer.encode('To be')]),8,greedy=True)
    result={'archive_sha256':expected['sha256'],'snapshot_count':len(verified),'checkpoint_sha256':file_hash(checkpoint),'cpu_load_and_generation':True,'text':tokenizer.decode(ids[0].tolist()),'label':'qualification_not_research','local_restore':str(restored)}
    atomic_json(output/'local_recovery_check.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();inspect(a.archive,a.manifest,a.output)
