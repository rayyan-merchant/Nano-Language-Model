"""Independent published ZIP, all checkpoint counters, sampling and CPU inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.bpe import encode_with_bytes
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.generate import generate
from nano_lm.evaluate import evaluation_tokens
from scripts.final.protocol import MODEL, BPE, PARAMETERS, TARGET, VALIDATION_STEPS


def sampling(payload,tokenizer,data):
    text=(data/'train.txt').read_bytes().decode('utf-8');ids,lengths=encode_with_bytes(tokenizer,text)
    dataset_manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(text.encode()).hexdigest()==dataset_manifest['partitions']['train']['sha256']
    assert payload['metadata']['resolved']['dataset_hash']==dataset_manifest['manifest_hash']
    offsets=np.concatenate(([0],np.cumsum(lengths)))
    difference=np.zeros(len(text.encode())+1,dtype=np.int64)
    generator=torch.Generator().manual_seed(42);exposure=0;input_bytes=0;minimum=10**9;maximum=0;prefix5k=None
    for micro_index in range(payload['step']*8):
        starts=torch.randint(len(ids)-128,(4,),generator=generator).numpy()
        low,high=offsets[starts+1],offsets[starts+129]
        exposure+=int((high-low).sum());np.add.at(difference,low,1);np.add.at(difference,high,-1)
        spans=offsets[starts+128]-offsets[starts]
        input_bytes+=int(spans.sum());minimum=min(minimum,int(spans.min()));maximum=max(maximum,int(spans.max()))
        if micro_index==5000*8-1:prefix5k=exposure
    seen=np.cumsum(difference[:-1])>0
    assert torch.equal(generator.get_state(),payload['rng']['sampling'])
    assert np.array_equal(seen,payload['training_state']['seen_bytes'])
    assert exposure==payload['training_state']['training_bytes']
    validation=(data/'validation.txt').read_bytes().decode('utf-8')
    seq,start,lens,_,_=evaluation_tokens(tokenizer,validation)
    _,prefix_lengths=encode_with_bytes(tokenizer,validation[:256])
    byte_offsets=np.concatenate(([0],np.cumsum(prefix_lengths+lens)))
    positions=np.arange(start,len(seq));context_spans=byte_offsets[positions]-byte_offsets[np.maximum(0,positions-128)]
    return {'status':'independent_private_sampling_RNG_source_byte_exposure_and_union_passed',
            'windows_per_update':32,'total_windows':payload['step']*32,'target_positions':payload['step']*4096,
            'original_source_byte_exposure':exposure,'unique_target_bytes':int(seen.sum()),
            'step5000_target_source_bytes':prefix5k,
            'sampled_training_input_context_bytes':{'mean':input_bytes/(payload['step']*32),'minimum':minimum,'maximum':maximum},
            'full_validation_preceding_context_bytes':{'mean':float(context_spans.mean()),'minimum':int(context_spans.min()),'maximum':int(context_spans.max())},
            'train_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'scope':'Only sampler replay; zero optimization; BPE offsets from frozen train-only tokenizer'}


def inspect(archive,manifest_path,out,data,development=False):
    started=time.perf_counter();torch.set_num_threads(1)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert file_hash(archive)==manifest['archive_sha256'] and archive.stat().st_size==manifest['archive_bytes']
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and set(names)==set(manifest['files']) and z.testzip() is None
        for name in names:
            path=Path(name);target=(out/name).resolve()
            if path.is_absolute() or ':' in name or '\\' in name or '..' in path.parts or out.resolve() not in target.parents:
                raise ValueError('Unsafe archive path')
        z.extractall(out)
    for name,sha in manifest['files'].items():assert file_hash(out/name)==sha
    if development:
        completion=json.loads((out/'development_completion.json').read_text())
        checkpoints=[('smoke',completion['smoke_checkpoint'])]
        assert manifest['run_kind']=='development' and completion['research_updates']==0
    else:
        completion=json.loads((out/'completion.json').read_text());assert manifest['run_kind']=='research'
        checkpoints=[('best',completion['best_checkpoint']),('final',completion['final_checkpoint'])]
    samples=[];checks=[]
    for policy,name in checkpoints:
        cp=out/name;payload,tok=read_checkpoint(cp)
        assert payload['config']==MODEL and tok.checksum==BPE and payload['metadata']['parameters']==PARAMETERS
        assert all(t.dtype==torch.float32 for t in payload['model'].values())
        settings=payload['metadata']['resolved']['training']
        for key,value in {'seed':42,'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.01,
                          'tokens_per_update':4096,'micro_batch_size':4,'gradient_clip':1.,
                          'precision':'float32','device':'cuda','conditioning_bytes':256,
                          'eval_mode':'sliding','eval_batch_size':32}.items():
            assert settings[key]==value
        if not development:
            assert settings['run_kind']=='research' and settings['eval_every']==250
            assert settings['checkpoint_every']==500 and settings['eval_max_bytes']==0
        model,_=load_for_inference(cp)
        count=8 if development else 64
        generated=generate(model,torch.tensor([tok.encode('ROMEO:\n')]),count,greedy=True,seed=42)
        samples.append({'policy':policy,'checkpoint':name,'sha256':file_hash(cp),'device':'cpu','greedy':True,
                        'seed':42,'max_new_tokens':count,'text':tok.decode(generated[0].tolist())})
        del model
    if not development:
        run=(out/completion['final_checkpoint']).parent.parent
        previous_evaluation_seconds=0.
        for step in VALIDATION_STEPS:
            payload,tok=read_checkpoint(run/f'checkpoints/step_{step:06d}.pt')
            state=payload['training_state'];history=state['history']
            assert payload['step']==state['step']==len(history)==step
            assert [r['step'] for r in history]==list(range(1,step+1))
            assert state['training_tokens']==step*4096 and state['amp_skipped_updates']==0
            assert {int(v['step'].item()) for v in payload['optimizer']['state'].values()}=={step}
            assert payload['scheduler']['last_epoch']==step
            assert all(r['accumulation']==8 and r['lr']==.0003 for r in history)
            validations=[r for r in history if 'validation' in r]
            assert [r['step'] for r in validations]==list(range(250,step+1,250))
            for row in validations:
                val=row['validation']
                assert val['scored_bytes']==55514 and val['scored_tokens']==24386 and val['conditioning_bytes']==256
                assert val['context_length']==128 and val['mode']=='sliding'
                assert math.isclose(val['bpb'],val['nll_sum']/(math.log(2)*55514),rel_tol=1e-12)
            checks.append({'step':step,'optimizer_and_scheduler_counters':step,'history_rows':len(history),
                           'measured_full_validation_seconds':state['eval_seconds']-previous_evaluation_seconds,
                           'cumulative_validation_seconds':state['eval_seconds'],'cumulative_training_seconds':state['train_seconds']})
            previous_evaluation_seconds=state['eval_seconds']
        assert payload['step']==TARGET
        sampler=sampling(payload,tok,data)
        events=[json.loads(x) for x in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        journal=[e['metric']['step'] for e in events if e['event']=='step']
        resumes=[e for e in events if e['event']=='resume'];failures=[e for e in events if e['event']=='failure']
        index=json.loads((out/'snapshot_index.json').read_text());assert len(index)==41
        assert sorted(m['step'] for m in index)==sorted(VALIDATION_STEPS+[10000])
        receipt={'checkpoint_counters':checks,'sampler':sampler,'snapshot_manifests':41,
                 'authoritative_updates':TARGET,'full_validations':40,'journal_recorded_update_attempts':len(journal),
                 'resume_events':resumes,'failure_events':failures,
                 'physical_execution_accounting':'Exactly10000 known physical updates only if complete sequential journal and no failures/resumes; otherwise reconcile attempts, unjournaled work remains unknown',
                 'uninterrupted_sequential_journal':journal==list(range(1,10001)) and not resumes and not failures}
    else:receipt={'research_updates':0,'smoke_step':1,'snapshot_files':'Native snapshot verification is performed independently in fresh CUDA recovery; ZIP contains hash inventory'}
    receipt.update(status='published_archive_size_SHA256_inventory_CRC_file_hashes_and_CPU_inference_passed',
                   archive_sha256=manifest['archive_sha256'],archive_bytes=manifest['archive_bytes'],
                   verified_file_count=len(manifest['files']),samples=samples,seconds=time.perf_counter()-started,
                   training_resume='CPU inference only; CUDA exact-state receipt is separately required')
    atomic_json(out/'independent_recovery.json',receipt)
    print(json.dumps({'status':receipt['status'],'files':len(manifest['files']),'samples':samples},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--development',action='store_true')
    a=p.parse_args();inspect(a.archive,a.manifest,a.output,a.dataset,a.development)
