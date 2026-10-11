"""Independently verify and recover a published trusted pilot archive."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate


def inspect(archive,manifest_path,out):
    began=time.perf_counter();manifest=json.loads(manifest_path.read_text())
    assert file_hash(archive)==manifest['archive_sha256'] and archive.stat().st_size==manifest['archive_bytes']
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest['files'])
        for name in z.namelist():
            target=(out/name).resolve()
            if out.resolve() not in target.parents:raise ValueError('Unsafe archive path')
        z.extractall(out)
    for name,checksum in manifest['files'].items():assert file_hash(out/name)==checksum,name
    completion=json.loads((out/'completion.json').read_text());torch.set_num_threads(1)
    final,_=read_checkpoint(out/completion['final_checkpoint']);h=final['training_state']['history']
    assert final['step']==5000 and len(h)==5000
    assert [r['step'] for r in h if 'validation' in r]==list(range(250,5001,250))
    samples=[]
    for label,key in [('best','best_checkpoint'),('final','final_checkpoint')]:
        path=out/completion[key];model,tokenizer=load_for_inference(path)
        result=generate(model,torch.tensor([tokenizer.encode('ROMEO:\n')]),64,greedy=True,seed=42)
        samples.append({'policy':label,'checkpoint':completion[key],'sha256':file_hash(path),'prompt':'ROMEO:\n','greedy':True,'seed':42,'max_new_tokens':64,'text':tokenizer.decode(result[0].tolist())})
    atomic_json(out/'independent_recovery.json',{'status':'published_archive_verified_and_best_final_CPU_inference_passed','run_id':completion['run_id'],'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],'verified_file_count':len(manifest['files']),'optimizer_updates':5000,'validation_count':20,'seconds':time.perf_counter()-began,'samples':samples,'training_resume':'CPU inference only; no CUDA RNG restore or additional updates attempted'})
    print(json.dumps({'status':'verified','run_id':completion['run_id'],'files':len(manifest['files']),'samples':samples},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();inspect(a.archive,a.manifest,a.output)
