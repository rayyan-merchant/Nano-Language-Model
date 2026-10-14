"""Independent B01 ZIP, variable stop/counters, sampler and CPU inference audit."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import math
from dataclasses import asdict
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate
from nano_lm.train import TrainSettings
from scripts.final.inspect import sampling
from scripts.balanced.protocol import MODEL, PARAMETERS
from scripts.balanced.stopping import replay


def inspect(archive,manifest_path,out,data,development=False):
    started=time.perf_counter();torch.set_num_threads(1)
    m=json.loads(Path(manifest_path).read_text(encoding='utf-8'));out=Path(out)
    assert file_hash(archive)==m['archive_sha256'] and Path(archive).stat().st_size==m['archive_bytes']
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and set(names)==set(m['files']) and z.testzip() is None
        for name in names:
            p=Path(name);target=(out/p).resolve()
            assert not p.is_absolute() and ':' not in name and '\\' not in name and '..' not in p.parts and out.resolve() in target.parents
        z.extractall(out)
    for name,sha in m['files'].items():assert file_hash(out/name)==sha
    checks=[];extra={}
    if development:
        c=json.loads((out/'development_completion.json').read_text(encoding='utf-8'))
        paths=[('development',out/c['smoke_checkpoint'])]
    else:
        c=json.loads((out/'completion.json').read_text(encoding='utf-8'));target=c['actual_completed_updates']
        run=(out/c['final_checkpoint']).parent.parent
        metadata=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        expected_settings=asdict(TrainSettings(device='cuda',run_kind='research',eval_every=250,checkpoint_every=500))
        assert metadata['resolved']['training']==expected_settings
        assert metadata['resolved']['experiment_id']=='B01' and metadata['resolved']['schedule']=='constant'
        assert metadata['resolved']['tokenizer_kind']=='bpe'
        assert metadata['resolved']['tokenizer_hash']=='2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55'
        for cp in sorted((run/'checkpoints').glob('step_*.pt')):
            payload,tok=read_checkpoint(cp);step=payload['step'];s=payload['training_state']
            assert payload['metadata']==metadata and tok.checksum==metadata['resolved']['tokenizer_hash']
            assert payload['config']==MODEL and payload['metadata']['parameters']==PARAMETERS
            assert len(s['history'])==step==int(cp.stem.split('_')[1])
            assert [r['step'] for r in s['history']]==list(range(1,step+1))
            assert {int(v['step'].item()) for v in payload['optimizer']['state'].values()}=={step}
            assert payload['scheduler']['last_epoch']==step and s['training_tokens']==step*4096 and s['amp_skipped_updates']==0
            assert all(r['accumulation']==8 and r['lr']==.0003 and math.isfinite(r['gradient_norm']) and math.isfinite(r['training_loss']) for r in s['history'])
            validations=[r for r in s['history'] if 'validation' in r]
            assert [r['step'] for r in validations]==list(range(250,step+1,250))
            for row in validations:
                v=row['validation'];assert (v['scored_bytes'],v['scored_tokens'],v['context_length'],v['conditioning_bytes'],v['mode'])==(55514,24386,128,256,'sliding')
                assert math.isclose(v['bpb'],v['nll_sum']/(math.log(2)*55514),rel_tol=1e-12)
            checks.append({'step':step,'optimizer_scheduler_counter':step,'history_rows':step})
        assert len(checks)==target//250 and step==target
        rule=replay(s['history']);assert rule.reason==c['stopping_reason'] and rule.records==c['patience_history'] and rule.best==c['absolute_best']
        events=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [e['metric']['step'] for e in events if e['event']=='step']==list(range(1,target+1))
        assert [e['step'] for e in events if e['event']=='resume']==list(range(250,target,250))
        assert not any(e['event']=='failure' for e in events)
        extra={'authoritative_updates':target,'known_physical_updates':target,'planned_resumes':target//250-1,
            'full_validations':target//250,'checkpoint_counters':checks,'stopping_rule_replay_passed':True,
            'sampler':sampling(payload,tok,Path(data)),'research_failures':0,'test_likelihood_evaluations':0}
        paths=[('best',out/c['best_checkpoint']),('final',out/c['final_checkpoint'])]
    samples=[]
    for policy,cp in paths:
        model,tok=load_for_inference(cp)
        assert model.config.to_dict()==MODEL and model.parameter_count()==PARAMETERS
        result=generate(model,torch.tensor([tok.encode('ROMEO:\n')]),64 if not development else 8,greedy=True)
        samples.append({'policy':policy,'checkpoint_sha256':file_hash(cp),'text':tok.decode(result[0].tolist()),
            'device':'cpu','greedy':True,'seed':42,'new_tokens':64 if not development else 8})
    receipt={'status':'B01_archive_CRC_safe_inventory_all_SHA256_and_CPU_inference_passed',
        'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'verified_file_count':len(m['files']),
        'samples':samples,'seconds':time.perf_counter()-started,'scope':'CPU inference/sampler replay only; no optimizer or test evaluation',**extra}
    atomic_json(out/'independent_recovery.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in {'samples','sampler','checkpoint_counters'}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['archive','manifest','output','dataset']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--development',action='store_true');a=p.parse_args()
    inspect(a.archive,a.manifest,a.output,a.dataset,a.development)
