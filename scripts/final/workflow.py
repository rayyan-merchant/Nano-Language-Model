"""Fail-closed F01 qualification wrapper; all optimization uses nano_lm.train."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.evaluate import evaluate
from nano_lm.generate import generate
from nano_lm.prepare import prepare, validate
from nano_lm.storage import export_run, verify_snapshot, restore_snapshot
from nano_lm.train import TrainSettings, train, environment
from nano_lm.verification import equal_nested
from scripts.final.protocol import configuration, provenance, PARAMETERS, BPE, TARGET, VALIDATION_STEPS
from scripts.final.audit import audit
from scripts.final.state import roundtrip, FIELDS


def qualified(root,out):
    env=environment(root)
    assert not env['git_dirty'] and env['git_commit']
    proof=provenance(root);config=configuration(root)
    assert env['python']=='3.11.14' and env['torch']=='2.5.1+cu124' and env['cuda']=='12.4' and env['gpu']=='Tesla T4'
    norm=lambda xs:sorted(x for x in xs if x and not x.startswith('-e '))
    assert norm(env['packages'])==norm((root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines())
    atomic_json(out/'source_provenance.json',{**proof,'F01_aggregate_code_hash':env['code_hash'],'clean_execution_commit':env['git_commit']})
    atomic_json(out/'environment.json',env)
    hardware={'kernel_id':os.environ['NANO_KERNEL_ID'],'training_device':'cuda:0',
              'session_budget_seconds':int(os.environ['NANO_SESSION_LIMIT_SECONDS']),
              'free_disk_bytes':shutil.disk_usage(out).free,'free_GPU_bytes':torch.cuda.mem_get_info(0)[0],
              'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory,
              'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    atomic_json(out/'hardware.json',hardware)
    return env,hardware


def archive(out,name,excluded=(),research=False):
    path=out.parent/name;began=time.perf_counter()
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*')
           if p.is_file() and not any(part in excluded for part in p.relative_to(out).parts)}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in files:z.write(out/name,name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and set(z.namelist())==set(files)
    manifest={'archive_file':path.name,'archive_bytes':path.stat().st_size,'archive_sha256':file_hash(path),
              'files':files,'archive_and_CRC_seconds':time.perf_counter()-began,
              'source_commit':json.loads((out/'environment.json').read_text())['git_commit'],
              'run_id':json.loads((out/'completion.json').read_text())['run_id'] if research else 'F01_development',
              'run_kind':'research' if research else 'development','research_updates':TARGET if research else 0}
    atomic_json(out.parent/'artifact_manifest.json',manifest)
    return manifest


def verify_archive(original):
    manifest=json.loads((original.parent/'artifact_manifest.json').read_text())
    path=original.parent/manifest['archive_file']
    assert path.stat().st_size==manifest['archive_bytes'] and file_hash(path)==manifest['archive_sha256']
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and set(names)==set(manifest['files']) and z.testzip() is None
        for name in names:
            p=Path(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name
    for name,sha in manifest['files'].items():assert file_hash(original/name)==sha
    return manifest


def development(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    env,hardware=qualified(root,out)
    data=Path('/tmp/F01-data');manifest=prepare(data,transport='http');validate(data)
    expected=json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    assert all(manifest[k]==expected[k] for k in ['source','split','partitions','tokenizers','manifest_hash'])
    assert manifest['tokenizers']['bpe']['hash']==BPE
    atomic_json(out/'dataset_validation.json',{'all_reference_fields_match':True,'manifest':manifest})
    atomic_json(out/'CUDA_architecture_audit.json',audit(root,data,'cuda'))
    subprocess.run([sys.executable,'-m','pytest','-q','--basetemp','/tmp/F01-pytest'],cwd=root,check=True)
    smoke=TrainSettings(device='cuda',run_kind='smoke',eval_every=1,checkpoint_every=1,eval_max_bytes=512)
    directory,checkpoint,_=train(root,data,'F01',smoke,1,out/'smoke',export_root=out/'smoke_snapshots')
    atomic_json(out/'CUDA_smoke_roundtrip.json',roundtrip(checkpoint,out/'roundtrip.pt','cuda'))
    torch.cuda.empty_cache()
    benchmark=TrainSettings(device='cuda',run_kind='benchmark',eval_every=10,checkpoint_every=10)
    began=time.perf_counter()
    _,cp,_=train(root,data,'F01',benchmark,10,out/'benchmark',export_root=out/'benchmark_snapshots')
    payload,_=read_checkpoint(cp);state=payload['training_state'];history=state['history']
    assert history[-1]['accumulation']==8 and history[-1]['validation']['scored_bytes']==55514
    assert history[-1]['validation']['scored_tokens']==24386 and history[-1]['validation']['context_length']==128
    warm_seconds=history[-1]['train_seconds']-history[1]['train_seconds']
    projected_train=10000*warm_seconds/8
    projected_validation=40*state['eval_seconds']
    # Storage holds 40 research checkpoints, up to two checkpoint copies per
    # snapshot (41), final ZIP, all development evidence, and a 2GiB margin.
    storage_required=170*cp.stat().st_size+2*2**30
    began_export=time.perf_counter();snap=export_run(directory,out/'development_snapshots');m=verify_snapshot(snap)
    export_seconds=time.perf_counter()-began_export
    snap_bytes=sum(p.stat().st_size for p in snap.rglob('*') if p.is_file())
    # Explicit conservative margin includes plotting, three generation grids,
    # one reserved final test, archive IO and platform publication estimate.
    projected_total=projected_train+projected_validation+1800
    feasibility={'label':'F01 development estimates, not research measurements','benchmark_updates':10,
                 'warmup_updates_excluded':2,'warmup_excluded_updates_per_second':8/warm_seconds,
                 'projected_10000_training_seconds':projected_train,'full_validation_seconds':state['eval_seconds'],
                 'projected_40_validation_seconds':projected_validation,'projected_pipeline_seconds_with_1800s_margin':projected_total,
                 'session_budget_seconds':hardware['session_budget_seconds'],'checkpoint_bytes':cp.stat().st_size,
                 'snapshot_bytes':snap_bytes,'snapshot_export_verify_seconds':export_seconds,
                 'benchmark_wall_seconds':time.perf_counter()-began,'peak_GPU_bytes':state['peak_gpu_bytes'],
                 'conservative_storage_required_bytes':storage_required,'available_disk_bytes':shutil.disk_usage(out).free,
                 'time_feasible':projected_total<hardware['session_budget_seconds'],
                 'storage_feasible':storage_required<shutil.disk_usage(out).free,
                 'publication_margin_is_estimate':True,'research_updates':0}
    atomic_json(out/'feasibility.json',feasibility)
    atomic_json(out/'development_completion.json',{'experiment':'F01','source_commit':env['git_commit'],
                'aggregate_code_hash':env['code_hash'],'smoke_checkpoint':str(checkpoint.relative_to(out)),
                'snapshot':str(snap.relative_to(out)),'snapshot_hash':m['hash'],'research_updates':0,
                'feasibility_passed':feasibility['time_feasible'] and feasibility['storage_feasible']})
    # Prove headless rendering before research, using authentic capped smoke.
    subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(directory),
                    '--output',str(out/'smoke_analysis'),'--device','cuda','--development-check'],check=True)
    index=[{'snapshot':p.relative_to(out).as_posix(),**verify_snapshot(p)}
           for folder in ['smoke_snapshots','benchmark_snapshots','development_snapshots'] for p in (out/folder).iterdir()]
    atomic_json(out/'snapshot_index.json',index)
    archive(out,'F01_development.zip',excluded=('smoke_snapshots','benchmark_snapshots','development_snapshots','roundtrip.pt'))
    print('F01 development published-output candidate ready; research updates ZERO',flush=True)


def development_recovery(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    env,_=qualified(root,out)
    completions=list(Path('/kaggle/input').rglob('development_completion.json'));assert len(completions)==1
    original=completions[0].parent;completion=json.loads(completions[0].read_text())
    assert completion['experiment']=='F01' and completion['source_commit']==env['git_commit']
    manifest=verify_archive(original)
    snapshot=original/completion['snapshot'];m=verify_snapshot(snapshot)
    assert m['hash']==completion['snapshot_hash']
    restored=restore_snapshot(snapshot,out/'restored');cp=restored/'checkpoints/step_000001.pt'
    receipt=roundtrip(cp,out/'fullstate_roundtrip.pt','cuda')
    data=Path('/tmp/F01-data');prepare(data,transport='http')
    payload,_=read_checkpoint(cp);settings=TrainSettings(**payload['metadata']['resolved']['training'])
    _,resumed,_=train(root,data,'F01',settings,2,out/'resumed',resume=cp)
    _,straight,_=train(root,data,'F01',settings,2,out/'uninterrupted')
    a,_=read_checkpoint(resumed);b,_=read_checkpoint(straight)
    equal={k:equal_nested(a[k],b[k]) for k in ['model','optimizer','scheduler','scaler','rng']}
    assert all(equal.values())
    for key in ['training_tokens','training_bytes','seen_bytes','step']:
        assert equal_nested(a['training_state'][key],b['training_state'][key])
    assert [r['training_loss'] for r in a['training_state']['history']]==[r['training_loss'] for r in b['training_state']['history']]
    receipt.update(status='published_F01_development_fresh_CUDA_full_state_and_continuity_passed',
                   source_commit=env['git_commit'],aggregate_code_hash=env['code_hash'],environment=env,
                   archive_sha256=manifest['archive_sha256'],verified_file_count=len(manifest['files']),
                   smoke_only_continuity_equal=equal,smoke_target_step=2,research_optimizer_updates=0,
                   feasibility=json.loads((original/'feasibility.json').read_text()),
                   snapshot_hash=m['hash'])
    atomic_json(out/'development_recovery.json',receipt)


def research(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);began=time.perf_counter()
    env,hardware=qualified(root,out)
    receipts=list(Path('/kaggle/input').rglob('development_recovery.json'));assert len(receipts)==1
    gate=json.loads(receipts[0].read_text())
    assert gate['status']=='published_F01_development_fresh_CUDA_full_state_and_continuity_passed'
    assert gate['source_commit']==env['git_commit'] and gate['aggregate_code_hash']==env['code_hash']
    assert gate['parameters']==PARAMETERS and gate['step_before']==gate['step_after']==1
    assert gate['additional_optimizer_updates']==gate['research_optimizer_updates']==0
    assert set(gate['exact_state'])==set(FIELDS) and all(gate['exact_state'].values())
    assert all(gate['smoke_only_continuity_equal'].values())
    assert gate['feasibility']['time_feasible'] and gate['feasibility']['storage_feasible']
    assert hardware['free_disk_bytes']>gate['feasibility']['conservative_storage_required_bytes']
    assert gate['feasibility']['projected_pipeline_seconds_with_1800s_margin']<hardware['session_budget_seconds']
    atomic_json(out/'independent_development_recovery_gate.json',gate)
    data=Path('/tmp/F01-data');manifest=prepare(data,transport='http');validate(data)
    reference=json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    assert all(manifest[k]==reference[k] for k in ['source','split','partitions','tokenizers','manifest_hash'])
    atomic_json(out/'dataset_validation.json',{'all_reference_fields_match':True,'manifest':manifest})
    atomic_json(out/'CUDA_architecture_audit.json',audit(root,data,'cuda'))
    # Use the actual CLI entrypoint. Hooks time original IO calls unchanged.
    import nano_lm.train as engine
    import nano_lm.storage as storage
    io=[];journal_seconds=[0.]
    def timed(name,function):
        def call(*args,**kwargs):
            t=time.perf_counter();result=function(*args,**kwargs)
            io.append({'operation':name,'seconds':time.perf_counter()-t,'reference':str(args[0])})
            return result
        return call
    original_save,original_export,original_event=engine.save_checkpoint,storage.export_run,engine.append_event
    engine.save_checkpoint=timed('checkpoint',original_save);storage.export_run=timed('export',original_export)
    def event(path,value):
        t=time.perf_counter();original_event(path,value);journal_seconds[0]+=time.perf_counter()-t
        if value['event']=='step' and value['metric']['step']%250==0:
            print('F01 RESEARCH PROGRESS',json.dumps(value['metric']),flush=True)
    engine.append_event=event
    argv=['nano_lm.train','--root',str(root),'--dataset',str(data),'--experiment','F01','--seed','42',
          '--steps','10000','--device','cuda','--precision','float32','--run-kind','research',
          '--tokens-per-update','4096','--micro-batch-size','4','--optimizer','AdamW','--learning-rate','0.0003',
          '--weight-decay','0.01','--gradient-clip','1.0','--eval-every','250','--checkpoint-every','500',
          '--conditioning-bytes','256','--eval-max-bytes','0','--eval-mode','sliding','--eval-batch-size','32',
          '--output',str(out/'runs'),'--export-root',str(out/'snapshots')]
    atomic_json(out/'execution_command.json',{'entrypoint':'python -m nano_lm.train','arguments':argv[1:],
                'fresh_initialization':True,'authorized_seed':42,'target':TARGET,'reserved_test_not_evaluated':True})
    previous=sys.argv;sys.argv=argv;cli_start=time.perf_counter()
    try:engine.main()
    finally:
        sys.argv=previous;engine.save_checkpoint=original_save;storage.export_run=original_export;engine.append_event=original_event
        atomic_json(out/'io_timing.json',{'operations':io,'journal_seconds':journal_seconds[0],
                    'CLI_wall_seconds':time.perf_counter()-cli_start,'scope':'Training, validation, original checkpoint/export/journal calls; excludes publication/download'})
    runs=list((out/'runs').iterdir());assert len(runs)==1;run=runs[0]
    checkpoint=run/'checkpoints/step_010000.pt';payload,_=read_checkpoint(checkpoint);state=payload['training_state']
    history=state['history'];val=[r for r in history if 'validation' in r]
    assert payload['step']==len(history)==TARGET and [r['step'] for r in history]==list(range(1,TARGET+1))
    assert [r['step'] for r in val]==VALIDATION_STEPS and state['training_tokens']==40960000
    assert state['amp_skipped_updates']==0 and all(r['accumulation']==8 and r['lr']==.0003 for r in history)
    assert all(r['validation']['scored_bytes']==55514 and r['validation']['scored_tokens']==24386
               and r['validation']['conditioning_bytes']==256 and r['validation']['context_length']==128
               and r['validation']['mode']=='sliding' for r in val)
    assert all((run/f'checkpoints/step_{step:06d}.pt').exists() for step in VALIDATION_STEPS)
    best=min(val,key=lambda r:r['validation']['bpb'])
    assert state['best_checkpoint']==f"step_{best['step']:06d}.pt"
    atomic_json(out/'completion.json',{'status':'10000_authoritative_updates_and_40_full_validations',
                'run_id':run.name,'source_commit':env['git_commit'],'code_hash':env['code_hash'],
                'config_hash':payload['metadata']['config_hash'],'final_checkpoint':str(checkpoint.relative_to(out)),
                'best_checkpoint':str((run/'checkpoints'/state['best_checkpoint']).relative_to(out)),
                'validation_steps':VALIDATION_STEPS,'parameters':PARAMETERS,
                'training_tokens':state['training_tokens'],'training_source_bytes':state['training_bytes']})
    # Freeze lowest validation BPB checkpoint before touching reserved test likelihood.
    chosen=run/'checkpoints'/state['best_checkpoint']
    freeze={'selection':'minimum canonical validation BPB across 40 scheduled points',
            'checkpoint':str(chosen.relative_to(out)),'checkpoint_sha256':file_hash(chosen),
            'best_step':best['step'],'best_validation':best['validation'],
            'tokenizer_hash':BPE,'configuration_hash':payload['metadata']['config_hash'],
            'generation_protocol_sha256':file_hash(root/'configs/generation_protocol.json'),
            'source_commit':env['git_commit'],'no_selection_changes_after_test':True}
    atomic_json(out/'selection_freeze.json',freeze)
    model,tok=load_for_inference(chosen);model.to('cuda');test_began=time.perf_counter()
    test=evaluate(model,tok,(data/'test.txt').read_bytes().decode('utf-8'),conditioning_bytes=256,batch_size=32,mode='sliding')
    assert test['scored_bytes']==55514 and test['context_length']==128
    atomic_json(out/'reserved_test_evaluation.json',{'label':'One final reserved held-out assessment after frozen validation selection',
                'checkpoint_sha256':freeze['checkpoint_sha256'],'metrics':test,'seconds':time.perf_counter()-test_began,
                'partition_hash':manifest['partitions']['test']['sha256'],'selection_frozen_before_evaluation':True})
    del model,payload;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(run),
                    '--output',str(out/'analysis'),'--device','cuda'],check=True)
    # Step5000 grid is an explicitly predeclared separate internal diagnostic.
    protocol=json.loads((root/'configs/generation_protocol.json').read_text());samples=[]
    cp5=run/'checkpoints/step_005000.pt';model,tok=load_for_inference(cp5);model.to('cuda')
    for prompt in protocol['prompts']:
        settings=[dict(greedy=True,temperature=1.,seed=42)]+[dict(greedy=False,temperature=t,seed=s)
                  for t in protocol['temperatures'] for s in protocol['sampling_seeds']]
        for setting in settings:
            result=generate(model,torch.tensor([tok.encode(prompt)],device='cuda'),256,**setting)
            samples.append({'checkpoint_policy':'step5000_internal_diagnostic','checkpoint':cp5.name,
                            'checkpoint_sha256':file_hash(cp5),'prompt':prompt,'max_new_tokens':256,
                            **setting,'text':tok.decode(result[0].tolist())})
    atomic_json(out/'step5000_generation.json',{'protocol':protocol,'samples':samples,'selection':'All50 slots, not checkpoint selection'})
    del model;torch.cuda.empty_cache()
    snapshots=[{'snapshot':p.name,**verify_snapshot(p)} for p in (out/'snapshots').iterdir()]
    assert len(snapshots)==41
    atomic_json(out/'snapshot_index.json',snapshots)
    atomic_json(out/'wall_time.json',{'pipeline_seconds_before_archive':time.perf_counter()-began,
                'scope':'Wrapper start through test and generations; excludes bootstrap, archive, publication and independent retrieval'})
    archive(out,'F01_seed42_final.zip',excluded=('snapshots',),research=True)


def research_recovery(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    env,_=qualified(root,out)
    paths=list(Path('/kaggle/input').rglob('completion.json'));assert len(paths)==1
    original=paths[0].parent;completion=json.loads(paths[0].read_text())
    assert completion['status']=='10000_authoritative_updates_and_40_full_validations'
    assert completion['source_commit']==env['git_commit'] and completion['code_hash']==env['code_hash']
    manifest=verify_archive(original)
    snapshots=[(p,verify_snapshot(p)) for p in (original/'snapshots').iterdir()];assert len(snapshots)==41
    assert sorted(m['step'] for _,m in snapshots)==sorted(VALIDATION_STEPS+[TARGET])
    snapshot,m=max((x for x in snapshots if x[1]['step']==TARGET),key=lambda x:len(x[1]['files']))
    restored=restore_snapshot(snapshot,out/'restored');checkpoint=restored/'checkpoints/step_010000.pt'
    receipt=roundtrip(checkpoint,out/'restored_step_010000.pt','cuda')
    payload,_=read_checkpoint(checkpoint)
    assert len(payload['training_state']['history'])==TARGET
    samples=[]
    for policy,name in [('best',payload['training_state']['best_checkpoint']),('final',checkpoint.name)]:
        cp=restored/'checkpoints'/name;model,tok=load_for_inference(cp);model.to('cuda')
        ids=generate(model,torch.tensor([tok.encode('ROMEO:\n')],device='cuda'),64,greedy=True)
        samples.append({'policy':policy,'checkpoint':name,'sha256':file_hash(cp),'text':tok.decode(ids[0].tolist()),
                        'greedy':True,'seed':42,'max_new_tokens':64,'device':'cuda'})
        del model;torch.cuda.empty_cache()
    receipt.update(status='published_F01_research_fresh_CUDA_full_state_passed_no_updates',
                   source_commit=env['git_commit'],environment=env,archive_sha256=manifest['archive_sha256'],
                   verified_file_count=len(manifest['files']),verified_snapshot_count=41,samples=samples,
                   authoritative_updates=TARGET,full_validations=40)
    atomic_json(out/'recovery.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['development','development-recovery','research','research-recovery'],required=True)
    a=p.parse_args()
    try:
        if a.mode=='development':development(a.root,a.output)
        elif a.mode=='development-recovery':development_recovery(a.root,a.output)
        elif a.mode=='research':research(a.root,a.output)
        else:research_recovery(a.root,a.output)
    except BaseException as e:
        if a.output.exists():atomic_json(a.output/'failure.json',{'type':type(e).__name__,'message':str(e),'mode':a.mode})
        raise
