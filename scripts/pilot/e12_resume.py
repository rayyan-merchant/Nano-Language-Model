"""Recover a durable E12 snapshot; continue the original run to total step 5000."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint
from nano_lm.prepare import prepare, validate
from nano_lm.storage import verify_snapshot, restore_snapshot
from e12_workflow import qualified, research, SOURCE, CORE
from e12_kaggle import state_restore


def select_snapshot(paths):
    verified = [(p, verify_snapshot(p)) for p in paths]
    if not verified or len({m['run_id'] for _, m in verified}) != 1:
        raise ValueError('Require one nonempty verified E12 snapshot lineage')
    latest, manifest = max(verified, key=lambda pair: (pair[1]['step'], len(pair[1]['files'])))
    if not 0 < manifest['step'] < 5000:
        raise ValueError('Resume requires an incomplete research checkpoint below 5000')
    return verified, latest, manifest


def preflight(root, out, store, saved):
    env = qualified(root)
    paths = [p.parent for p in saved.rglob('export_manifest.json') if 'research_snapshots' in p.parts]
    verified, latest, manifest = select_snapshot(paths)
    out.mkdir(parents=True, exist_ok=False)
    store.mkdir(parents=True, exist_ok=False)
    for source, checksum in verified:
        destination = store/'research_snapshots'/source.name
        shutil.copytree(source, destination)
        assert verify_snapshot(destination) == checksum
    run = restore_snapshot(latest, out/'runs')
    cp = max((run/'checkpoints').glob('step_*.pt'))
    payload, _ = read_checkpoint(cp)
    assert payload['step'] == manifest['step']
    assert payload['metadata']['parameters'] == 1804800
    assert payload['metadata']['resolved']['code_hash'] == CORE
    from nano_lm.config import load_experiment
    config,_,variant=load_experiment(root,'E12',65)
    assert payload['config']==config.to_dict() and variant['model_overrides']=={'context_length':256}
    assert payload['metadata']['resolved']['experiment_id']=='E12'
    assert all(row['accumulation']==4 for row in payload['training_state']['history'])
    assert payload['config']['n_heads'] == 6 and payload['config']['context_length']==256 and payload['config']['activation'] == 'gelu'
    state = payload['training_state']
    assert [r['step'] for r in state['history']] == list(range(1, payload['step']+1))
    assert [r['step'] for r in state['history'] if 'validation' in r] == list(range(250,payload['step']+1,250))
    assert payload['scheduler']['last_epoch'] == payload['step']
    assert all(int(v['step']) == payload['step'] for v in payload['optimizer']['state'].values())
    recovered = state_restore(cp, out/'resume_roundtrip.pt')
    assert all(recovered['exact_state'].values()) and recovered['additional_optimizer_updates'] == 0
    # Preserve any saved attempts beyond the durable checkpoint as separate evidence.
    original = saved/'research'
    attempted = original/'runs'/run.name/'events.jsonl'
    if attempted.exists():
        events = [json.loads(line) for line in attempted.read_text().splitlines()]
        atomic_json(out/'original_saved_attempts.json',{'source_sha256':file_hash(attempted),'committed_resume_step':payload['step'],'events':events,'scope':'Saved original journal, including any attempts beyond the restored checkpoint. Never merged or truncated to manufacture continuation.'})
    for source in original.glob('*.json'):
        if source.name not in {'completion.json','io_timing.json','preflight.json','environment.json','hardware.json','execution.json'}:
            shutil.copyfile(source, out/source.name)
    if (original/'execution.json').exists():
        shutil.copyfile(original/'execution.json',out/'original_execution.json')
    actual = prepare(out/'data'); validate(out/'data')
    assert actual == json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    assert shutil.disk_usage(out).free > 5*2**30
    atomic_json(out/'environment.json',env)
    atomic_json(out/'hardware.json',{'gpu_count':torch.cuda.device_count(),'training_device':'cuda:0','nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'local_disk_free_bytes':shutil.disk_usage(out).free,'devices':[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})
    receipt = {'status':'hash_verified_full_state_CUDA_restoration_passed','run_id':run.name,'checkpoint_step':payload['step'],'checkpoint_sha256':file_hash(cp),'snapshot_hash':manifest['hash'],'snapshot_count':len(verified),'recovery':recovered,'additional_optimizer_updates':0,'interruption':'Previous draft session was off; later volatile progress unavailable and unclaimed. Continue newest durable checkpoint with original run ID, complete optimization and RNG state. Prior checkpoint/export overhead was not durably recorded.'}
    atomic_json(out/'resume_preflight.json',receipt)
    atomic_json(out/'preflight.json',{'status':'passed','platform':'Kaggle','resume':receipt})
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['preflight','resume'])
    for name in ['root','output','store']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--saved',type=Path)
    args=parser.parse_args()
    if args.phase == 'preflight':
        preflight(args.root,args.output,args.store,args.saved)
    else:
        receipt=json.loads((args.output/'resume_preflight.json').read_text())
        cp=args.output/'runs'/receipt['run_id']/'checkpoints'/f"step_{receipt['checkpoint_step']:06d}.pt"
        research(args.root,args.output,args.store,resume=cp)
