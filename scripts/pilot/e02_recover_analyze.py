"""Verify durable E02 snapshots and analyze genuine weights without training."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.bpe import encode_with_bytes, from_artifact
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate
from nano_lm.storage import restore_snapshot, verify_snapshot, snapshot_file
from nano_lm.train import environment

CORE = 'cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'


def recover(root, store, out, analyzer):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = environment(root)
    assert not env['git_dirty'] and env['code_hash'] == CORE
    assert env['python'] == '3.11.14' and env['torch'] == '2.5.1+cu124'
    assert env['cuda'] == '12.4' and env['gpu'] == 'Tesla T4'
    expected = (root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines()
    norm = lambda x: sorted(v for v in x if v and not v.startswith('-e '))
    assert norm(expected) == norm(env['packages'])
    atomic_json(out/'recovery_environment.json', env)
    began = time.perf_counter()
    snapshots = sorted((store/'research_snapshots').iterdir())
    manifests = {p: verify_snapshot(p) for p in snapshots}
    assert sorted(set(m['step'] for m in manifests.values())) == list(range(250, 5001, 250))
    latest = max(snapshots, key=lambda p: (manifests[p]['step'], len(manifests[p]['files'])))
    run = restore_snapshot(latest, out/'runs')
    # Rehydrate historical states solely from checksum-verified immutable snapshots.
    for snapshot, manifest in manifests.items():
        for relative, checksum in manifest['files'].items():
            if relative.startswith('checkpoints/'):
                destination = run/relative
                if destination.exists():
                    assert file_hash(destination) == checksum
                else:
                    shutil.copyfile(snapshot/relative, destination)
                    assert file_hash(destination) == checksum
    final, tokenizer = read_checkpoint(run/'checkpoints/step_005000.pt')
    metadata, state = final['metadata'], final['training_state']
    settings = metadata['resolved']['training']
    assert final['step'] == 5000 and metadata['resolved']['experiment_id'] == 'E02'
    assert settings['seed'] == 42 and settings['precision'] == 'float32'
    assert metadata['parameters'] == 2173056 and metadata['resolved']['code_hash'] == CORE
    history = state['history']
    validations = [r for r in history if 'validation' in r]
    assert [r['step'] for r in history] == list(range(1, 5001))
    assert [r['step'] for r in validations] == list(range(250, 5001, 250))
    assert state['training_tokens'] == 20480000 and state['amp_skipped_updates'] == 0
    assert all(r['lr'] == .0003 and r['accumulation'] == 8 for r in history)
    checks = []
    for path in sorted((run/'checkpoints').glob('step_*.pt')):
        payload, _ = read_checkpoint(path)
        step = payload['step']
        assert payload['metadata'] == metadata and len(payload['training_state']['history']) == step
        assert payload['training_state']['training_tokens'] == step*4096
        counters = sorted(set(int(v['step'].item()) for v in payload['optimizer']['state'].values()))
        assert counters == [step] and payload['scheduler']['last_epoch'] == step
        checks.append({'file':path.name, 'step':step, 'optimizer_counters':counters,
                       'scheduler_epoch':payload['scheduler']['last_epoch'], 'sha256':file_hash(path)})
    assert len(checks) == 20
    for row in validations:
        v = row['validation']
        assert v['scored_bytes'] == 55514 and v['conditioning_bytes'] == 256
        assert v['evaluated_text_bytes'] == 55770 and v['mode'] == 'sliding'
        assert math.isclose(v['bpb'], v['nll_sum']/(math.log(2)*55514), rel_tol=1e-12)
        assert math.isclose(v['loss'], v['nll_sum']/v['scored_tokens'], rel_tol=1e-12)
    best = min(validations, key=lambda r:r['validation']['loss'])
    assert state['best_checkpoint'] == f"step_{best['step']:06d}.pt"
    events = [json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='step' for e in events) == 5000
    failures = [e for e in events if e['event'] in {'failure','resume'}]
    recovery_samples = []
    for label, checkpoint in [('final','step_005000.pt'), ('best',state['best_checkpoint'])]:
        restored = restore_snapshot(latest, out/(label+'_independent_restore'))
        model, tok = load_for_inference(restored/'checkpoints'/checkpoint)
        model.to('cuda')
        ids = torch.tensor([tok.encode('To be')], device='cuda')
        generated = generate(model, ids, 16, greedy=True)
        recovery_samples.append({'label':label, 'checkpoint':checkpoint,
                                 'text':tok.decode(generated[0].tolist())})
        del model
        torch.cuda.empty_cache()
    evidence = {'status':'5000_updates_20_validations_20_checkpoint_counters_verified',
                'run_id':run.name, 'snapshot_count':len(snapshots), 'checkpoints':checks,
                'validation_steps':[r['step'] for r in validations], 'journal_steps':5000,
                'training_tokens':state['training_tokens'], 'training_source_bytes':state['training_bytes'],
                'unique_target_bytes':history[-1]['unique_target_bytes'], 'journal_failures_or_resumes':failures,
                'independent_recovered_inference':recovery_samples,
                'snapshot_verification_restore_audit_seconds':time.perf_counter()-began,
                'runtime_note':'Original training completed before original runtime disappeared; this fresh runtime performs zero optimizer updates.'}
    atomic_json(out/'checkpoint_audit.json', evidence)
    atomic_json(out/'snapshot_index.json', [{'directory':p.name, **m} for p,m in manifests.items()])
    # Original-text context coverage from the actual tokenizer, with no test evaluation.
    subprocess.run([sys.executable, '-m', 'nano_lm.prepare', '--output', str(out/'data')], cwd=root, check=True)
    data = out/'data'
    assert json.loads((data/'manifest.json').read_text()) == json.loads((run/'dataset_manifest.json').read_text())
    train_text = (data/'train.txt').read_text()
    validation_text = (data/'validation.txt').read_text()
    _, lengths = encode_with_bytes(tokenizer, train_text)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    spans = offsets[128:len(lengths)]-offsets[:len(lengths)-128]
    prefix = validation_text.encode()[:256].decode()
    suffix = validation_text.encode()[256:].decode()
    _, a = encode_with_bytes(tokenizer, prefix)
    _, b = encode_with_bytes(tokenizer, suffix)
    boundaries = np.concatenate(([0], np.cumsum(a+b)))
    contexts = np.array([boundaries[i]-boundaries[max(0,i-128)] for i in range(len(a),len(a)+len(b))])
    stats = lambda v: {'mean':float(np.mean(v)), 'min':int(np.min(v)), 'max':int(np.max(v)), 'median':float(np.median(v))}
    atomic_json(out/'source_byte_context.json', {'training_128_token_context_bytes':stats(spans),
                'validation_scored_target_preceding_context_bytes':stats(contexts),
                'training_exposure_bytes_per_predicted_token':state['training_bytes']/state['training_tokens'],
                'training_unique_byte_coverage_fraction':history[-1]['unique_target_bytes']/len(train_text.encode()),
                'E01_128_token_full_context_bytes':128, 'policy':'Actual BPE byte lengths; exhaustive possible training windows and canonical separately tokenized validation prefix/suffix. No test likelihood.'})
    subprocess.run([sys.executable,str(analyzer),'--root',str(root),'--run',str(run),'--output',str(out/'analysis'),'--device','cuda'], cwd=root, check=True)
    for name in ['analysis','checkpoint_audit.json','snapshot_index.json','recovery_environment.json','source_byte_context.json']:
        src = out/name
        dst = store/name
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copyfile(src,dst)
    archive = store/'E02_seed42_complete.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for folder in [run, out/'analysis']:
            for p in folder.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(out).as_posix())
        for p in out.glob('*.json'): z.write(p,p.name)
    finalize_archive(store, out)
    print('E02 RECOVERY AND ANALYSIS COMPLETE', json.dumps(evidence), flush=True)


def finalize_archive(store, out):
    """Check every archived file; publish using Drive-compatible staged rename."""
    archive = store/'E02_seed42_complete.zip'
    run = next((out/'runs').iterdir())
    metadata = json.loads((run/'metadata.json').read_text())
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        assert z.testzip() is None
        import hashlib
        hashes = {n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
        assert all(file_hash(out/n) == checksum for n,checksum in hashes.items())
    manifest = {'run_id':run.name,'archive_sha256':file_hash(archive),
                'archive_bytes':archive.stat().st_size,'files':hashes,
                'archive_crc_and_all_entry_source_hashes_verified':True,
                'storage':str(store),'source_commit':metadata['environment']['git_commit']}
    snapshot_file(store/'artifact_manifest.json', (json.dumps(manifest,indent=2)+'\n').encode())
    print('ARCHIVE MANIFEST VERIFIED AND PUBLISHED',manifest['archive_sha256'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['root','store','output','analyzer']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args()
    if a.finalize_only: finalize_archive(a.store,a.output)
    else: recover(a.root,a.store,a.output,a.analyzer)
