"""Independently reconstruct downloaded E02 parts, audit hashes and infer on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile
import torch
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def verify(evidence):
    began=time.perf_counter()
    parts=json.loads((evidence/'archive_parts.json').read_text())
    manifest=json.loads((evidence/'artifact_manifest.json').read_text())
    for part in parts['parts']:
        path=evidence/part['name']
        assert path.stat().st_size==part['bytes'] and checksum(path)==part['sha256']
    archive=evidence/'E02_seed42_complete.zip'
    if not archive.exists():
        with archive.open('xb') as out:
            for part in parts['parts']:
                with (evidence/part['name']).open('rb') as source:shutil.copyfileobj(source,out,2**20)
    assert archive.stat().st_size==manifest['archive_bytes']==parts['archive_bytes']
    assert checksum(archive)==manifest['archive_sha256']==parts['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and set(names)==set(manifest['files'])
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        assert z.testzip() is None
        for name in names:
            assert hashlib.sha256(z.read(name)).hexdigest()==manifest['files'][name]
        run_prefix='runs/'+manifest['run_id']+'/'
        metadata=json.loads(z.read(run_prefix+'metadata.json'))
        assert metadata['resolved']['experiment_id']=='E02' and metadata['parameters']==2173056
        expected={'seed':42,'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.01,
                  'tokens_per_update':4096,'micro_batch_size':4,'gradient_clip':1.,
                  'precision':'float32','device':'cuda','eval_every':250,'checkpoint_every':500,
                  'conditioning_bytes':256,'eval_max_bytes':0,'eval_mode':'sliding','eval_batch_size':32,'run_kind':'research'}
        assert metadata['resolved']['training']==expected
        state_paths=[n for n in names if n.startswith(run_prefix+'checkpoints/')]
        assert len(state_paths)==20
        audit=json.loads((evidence/'checkpoint_audit.json').read_text())
        for row in audit['checkpoints']:
            assert manifest['files'][run_prefix+'checkpoints/'+row['file']]==row['sha256']
        for name in names:
            if '/checkpoints/' in name and not name.endswith('step_005000.pt'):continue
            destination=evidence/'extracted'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(z.read(name))
        (evidence/'training_environment.json').write_text(json.dumps(metadata['environment'],indent=2))
        (evidence/'figures').mkdir(exist_ok=True)
        for name in names:
            if name.startswith('analysis/figures/') and name.endswith('.png'):
                (evidence/'figures'/Path(name).name).write_bytes(z.read(name))
    checkpoint=evidence/'extracted'/run_prefix/'checkpoints/step_005000.pt'
    payload,tokenizer=read_checkpoint(checkpoint)
    state=payload['training_state'];history=state['history']
    assert payload['step']==5000 and len(history)==5000
    assert state['training_tokens']==20480000 and state['best_checkpoint']=='step_005000.pt'
    assert [r['step'] for r in history if 'validation' in r]==list(range(250,5001,250))
    counters=sorted(set(int(x['step'].item()) for x in payload['optimizer']['state'].values()))
    assert counters==[5000] and payload['scheduler']['last_epoch']==5000
    summary=json.loads((evidence/'summary.json').read_text())
    assert history[-1]['validation']==summary['final_validation']
    generation=json.loads((evidence/'generation.json').read_text())
    samples=generation['samples'];assert len(samples)==100
    best=[x for x in samples if x['checkpoint_policy']=='best']
    final=[x for x in samples if x['checkpoint_policy']=='final']
    assert len(best)==len(final)==50 and all(a['text']==b['text'] for a,b in zip(best,final))
    assert all(x['checkpoint_sha256']==checksum(checkpoint) for x in samples)
    torch.set_num_threads(1)
    model,tok=load_for_inference(checkpoint)
    ids=torch.tensor([tok.encode('To be')])
    result=generate(model,ids,16,greedy=True)
    text=tok.decode(result[0].tolist())
    value={'status':'independent_download_reconstruction_and_CPU_inference_passed',
           'archive_sha256':checksum(archive),'archive_bytes':archive.stat().st_size,
           'verified_parts':parts['parts'],'archive_entries':len(names),'all_archive_file_hashes_match':True,
           'zip_crc_paths_duplicates_passed':True,'checkpoint_files':len(state_paths),
           'final_checkpoint_sha256':checksum(checkpoint),'best_equals_final':True,
           'final_optimizer_counters':counters,'final_scheduler_epoch':payload['scheduler']['last_epoch'],
           'history_updates':len(history),'validation_measurements':20,'full_generation_records':100,
           'CPU_inference':{'prompt':'To be','max_new_tokens':16,'greedy':True,'text':text,
                           'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu',
                           'scope':'Inference migration only; no training continuation on this newer framework'},
           'verification_seconds':time.perf_counter()-began}
    (evidence/'independent_verification.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    verify(p.parse_args().evidence)
