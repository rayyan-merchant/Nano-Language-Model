"""Independently retrieve/reconstruct E07 bytes and load genuine best/final models."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile
import torch
from nano_lm.storage import verify_snapshot
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def verify(evidence):
    started=time.perf_counter()
    parts=json.loads((evidence/'archive_parts.json').read_text())
    manifest=json.loads((evidence/'artifact_manifest.json').read_text())
    assert [p['name'] for p in parts['parts']]==[f'part_{i:02d}.bin' for i in range(len(parts['parts']))]
    for part in parts['parts']:
        path=evidence/part['name']
        assert path.stat().st_size==part['bytes'] and checksum(path)==part['sha256']
    archive=evidence/parts['archive']
    if not archive.exists():
        with archive.open('xb') as out:
            for part in parts['parts']:
                with (evidence/part['name']).open('rb') as f:shutil.copyfileobj(f,out,2**20)
    assert archive.stat().st_size==parts['archive_bytes']==manifest['archive_bytes']
    assert checksum(archive)==parts['archive_sha256']==manifest['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and set(names)==set(manifest['files'])
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        assert z.testzip() is None
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==manifest['files'][n]
        prefix=manifest['run_prefix'];metadata=json.loads(z.read(prefix+'metadata.json'))
        assert metadata['resolved']['experiment_id']=='E07' and metadata['parameters']==1804800
        assert metadata['resolved']['model']['activation']=='gelu' and metadata['resolved']['model']['n_heads']==8 and metadata['resolved']['model']['position']=='sinusoidal'
        expected={'seed':42,'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.01,
                  'tokens_per_update':4096,'micro_batch_size':4,'gradient_clip':1.,
                  'precision':'float32','device':'cuda','eval_every':250,'checkpoint_every':500,
                  'conditioning_bytes':256,'eval_max_bytes':0,'eval_mode':'sliding','eval_batch_size':32,'run_kind':'research'}
        assert metadata['resolved']['training']==expected
        checkpoint_names=[n for n in names if n.startswith(prefix+'checkpoints/')]
        assert sorted(Path(n).name for n in checkpoint_names)==[f'step_{s:06d}.pt' for s in range(250,5001,250)]
        # Extract every entry after path/CRC/hash checks, then audit actual optimizer state.
        extracted=evidence/'extracted';extracted.mkdir(exist_ok=False)
        z.extractall(extracted)
    # Reconstruct all immutable snapshots from their exact metadata and canonical weight files.
    reconstructed=evidence/'reconstructed_snapshots';reconstructed.mkdir(exist_ok=False)
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
    assert len(manifests)==21 and sorted(set(m['step'] for m in manifests))==list(range(250,5001,250))
    final_path=extracted/prefix/'checkpoints/step_005000.pt'
    final,_=read_checkpoint(final_path);state=final['training_state'];history=state['history']
    assert final['step']==5000 and [r['step'] for r in history]==list(range(1,5001))
    assert state['training_tokens']==state['training_bytes']==20480000 and state['amp_skipped_updates']==0
    assert [r['step'] for r in history if 'validation' in r]==list(range(250,5001,250))
    checks=[]
    for name in checkpoint_names:
        payload,_=read_checkpoint(extracted/name);step=payload['step']
        counters=sorted(set(int(v['step'].item()) for v in payload['optimizer']['state'].values()))
        assert counters==[step] and payload['scheduler']['last_epoch']==step
        assert payload['metadata']==final['metadata'] and len(payload['training_state']['history'])==step
        checks.append({'checkpoint':Path(name).name,'step':step,'optimizer_counters':counters,'sha256':checksum(extracted/name)})
    generation=json.loads((extracted/'analysis/generation.json').read_text())
    assert len(generation['samples'])==100 and not generation['development_check']
    protocol_path=Path(__file__).resolve().parents[2]/'configs/generation_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert generation['protocol']==protocol and generation['protocol_source_sha256']==checksum(protocol_path)
    expected_grid={(prompt,True,1.0,42) for prompt in protocol['prompts']}
    expected_grid|={(prompt,False,temp,seed) for prompt in protocol['prompts'] for temp in protocol['temperatures'] for seed in protocol['sampling_seeds']}
    policies={policy:[x for x in generation['samples'] if x['checkpoint_policy']==policy] for policy in ['best','final']}
    for policy,rows in policies.items():
        assert len(rows)==50 and {(x['prompt'],x['greedy'],x['temperature'],x['seed']) for x in rows}==expected_grid
        assert all(x['max_new_tokens']==256 and x['generation_device']=='cuda' for x in rows)
    identical_policies=state['best_checkpoint']=='step_005000.pt'
    if identical_policies:
        assert all(x['text']==y['text'] for x,y in zip(policies['best'],policies['final']))
    for sample in generation['samples']:
        assert checksum(extracted/prefix/'checkpoints'/sample['checkpoint'])==sample['checkpoint_sha256']
    torch.set_num_threads(1);samples=[]
    for policy,name in [('best',state['best_checkpoint']),('final','step_005000.pt')]:
        path=extracted/prefix/'checkpoints'/name;model,tok=load_for_inference(path)
        generated=generate(model,torch.tensor([tok.encode('ROMEO:\n')]),64,greedy=True,seed=42)
        samples.append({'policy':policy,'checkpoint':name,'text':tok.decode(generated[0].tolist()),'sha256':checksum(path)})
    receipt={'status':'independent_download_reconstruction_all_hashes_and_CPU_inference_passed',
             'archive_sha256':checksum(archive),'archive_bytes':archive.stat().st_size,'verified_parts':parts['parts'],
             'archive_entries':len(names),'all_archive_hashes_and_CRC_passed':True,'checkpoint_count':len(checks),'checks':checks,'independently_reconstructed_verified_snapshots':len(manifests),
             'completed_optimizer_updates':5000,'full_validations':20,'generation_records':100,
             'best_equals_final':identical_policies,'duplicate_policy_outputs_verified':identical_policies,'fixed_protocol_50_settings_per_policy_verified':True,'CPU_inference':samples,
             'CPU_environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
             'additional_optimizer_updates':0,'scope':'CPU inference migration; compatible exact CUDA optimizer/RNG restoration audited separately.',
             'seconds_excluding_download':time.perf_counter()-started}
    (evidence/'independent_verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    verify(p.parse_args().evidence)
