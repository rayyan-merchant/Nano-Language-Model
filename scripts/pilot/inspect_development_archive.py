"""Independent local E14 published-development integrity and CPU inference."""
import argparse
import json
from pathlib import Path
import zipfile
import torch
from nano_lm.artifacts import atomic_json,file_hash
from nano_lm.checkpoint import read_checkpoint,load_for_inference
from nano_lm.generate import generate
from nano_lm.storage import verify_snapshot


def inspect(archive,manifest_path,out):
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert m['run_kind']=='development' and m['research_updates']==0
    assert file_hash(archive)==m['archive_sha256'] and archive.stat().st_size==m['archive_bytes']
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(m['files'])
        assert all(out.resolve() in (out/n).resolve().parents for n in z.namelist())
        z.extractall(out)
    for name,sha in m['files'].items():assert file_hash(out/name)==sha
    c=json.loads((out/'development_completion.json').read_text())
    assert c['experiment']=='E14' and c['step']==1 and c['research_updates']==0
    assert verify_snapshot(out/c['snapshot'])['hash']==c['snapshot_hash']
    path=out/c['checkpoint'];payload,_=read_checkpoint(path)
    assert payload['step']==1 and payload['metadata']['parameters']==2808960
    torch.set_num_threads(1);model,tok=load_for_inference(path)
    text=tok.decode(generate(model,torch.tensor([tok.encode('ROMEO:\n')]),4,greedy=True)[0].tolist())
    atomic_json(out/'independent_development_recovery.json',{'status':'published_development_archive_integrity_snapshot_and_CPU_inference_passed',
        'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'verified_file_count':len(m['files']),
        'checkpoint_sha256':file_hash(path),'source_commit':c['source_commit'],'optimizer_step':1,
        'genuine_CPU_text':text,'additional_optimizer_updates':0,'scope':'CPU inference only; full CUDA state restoration verified separately'})
    print(json.dumps({'files':len(m['files']),'archive_sha256':m['archive_sha256'],'CPU_text':text}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();inspect(a.archive,a.manifest,a.output)
