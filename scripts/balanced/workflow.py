"""Fail-closed B01 staged execution; original engine owns every optimizer update."""
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
from nano_lm.artifacts import atomic_json, file_hash, digest
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.storage import export_run, restore_snapshot, verify_snapshot
from nano_lm.train import TrainSettings, train, environment
from nano_lm.generate import generate
from scripts.final.workflow import verify_archive
from scripts.balanced.protocol import configuration, provenance, PARAMETERS, MODEL, MAXIMUM, STOP_POLICY, BPE
from scripts.balanced.data import prepare
from scripts.balanced.audit import audit
from scripts.balanced.continuation import verify as continuation
from scripts.balanced.state import roundtrip, compare_trajectory
from scripts.balanced.stopping import replay, StoppingRule
from scripts.balanced.analysis import analyze


def qualified(root,out):
    env=environment(root);configuration(root);proof=provenance(root)
    assert not env['git_dirty'] and env['git_commit']
    assert (env['python'],env['torch'],env['cuda'],env['gpu'])==('3.11.14','2.5.1+cu124','12.4','Tesla T4')
    norm=lambda rows: sorted(r for r in rows if r and not r.startswith('-e '))
    assert norm(env['packages'])==norm((root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text(encoding='utf-8').splitlines())
    atomic_json(out/'environment.json',env)
    atomic_json(out/'source_provenance.json',{**proof,'B01_aggregate_code_hash':env['code_hash'],'clean_execution_commit':env['git_commit']})
    hardware={'kernel_id':os.environ['NANO_KERNEL_ID'],'device':'cuda:0',
        'session_budget_seconds':int(os.environ['NANO_SESSION_LIMIT_SECONDS']),
        'free_disk_bytes':shutil.disk_usage(out).free,'free_GPU_bytes':torch.cuda.mem_get_info(0)[0],
        'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    atomic_json(out/'hardware.json',hardware)
    return env,hardware


def archive(out,name,research=False):
    started=time.perf_counter();path=out.parent/name
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file()
           and 'snapshots' not in p.relative_to(out).parts and p.name!='roundtrip.pt'}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for relative in files:z.write(out/relative,relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None and set(z.namelist())==set(files)
    completion=json.loads((out/('completion.json' if research else 'development_completion.json')).read_text(encoding='utf-8'))
    atomic_json(out.parent/'artifact_manifest.json',dict(archive_file=name,archive_bytes=path.stat().st_size,
        archive_sha256=file_hash(path),files=files,archive_and_CRC_seconds=time.perf_counter()-started,
        source_commit=completion['source_commit'],run_kind='research' if research else 'development',
        research_updates=completion['actual_completed_updates'] if research else 0))


def development(root,out):
    out.mkdir(parents=True,exist_ok=False);env,hardware=qualified(root,out)
    data=Path('/tmp/B01-data');manifest=prepare(root,data)
    atomic_json(out/'dataset_validation.json',{'manifest':manifest,'BPE_refitted':False})
    atomic_json(out/'CUDA_architecture_audit.json',audit(root,data,'cuda'))
    subprocess.run([sys.executable,'-m','pytest','-q','--basetemp','/tmp/B01-pytest'],cwd=root,check=True)
    result=continuation(root,data,out/'continuation','cuda',True)
    partial=next((out/'continuation/segmented').iterdir())
    cp=partial/'checkpoints/step_000002.pt';snapshot=export_run(partial,out/'snapshots')
    torch.cuda.empty_cache()
    settings=TrainSettings(device='cuda',run_kind='benchmark',eval_every=10,checkpoint_every=10)
    _,benchmark,_=train(root,data,'B01',settings,10,out/'benchmark')
    payload,_=read_checkpoint(benchmark);s=payload['training_state'];h=s['history']
    seconds=h[-1]['train_seconds']-h[1]['train_seconds']
    export_start=time.perf_counter();snap=export_run(benchmark.parent.parent,out/'snapshots');verify_snapshot(snap)
    storage=128*benchmark.stat().st_size+3*2**30
    estimate=MAXIMUM*seconds/8+20*s['eval_seconds']+1800
    feasibility={'label':'development-only maximum-budget estimates','warmup_updates_excluded':2,
        'updates_per_second':8/seconds,'full_validation_seconds':s['eval_seconds'],
        'projected_maximum_training_seconds':MAXIMUM*seconds/8,'projected_pipeline_with_margin_seconds':estimate,
        'checkpoint_bytes':benchmark.stat().st_size,'peak_GPU_bytes':s['peak_gpu_bytes'],
        'snapshot_export_verify_seconds':time.perf_counter()-export_start,
        'storage_required_bytes':storage,'available_disk_bytes':shutil.disk_usage(out).free,
        'time_feasible':estimate<hardware['session_budget_seconds'],'storage_feasible':storage<shutil.disk_usage(out).free}
    atomic_json(out/'feasibility.json',feasibility)
    analyze(root,partial,out/'smoke_analysis','cuda',True)
    atomic_json(out/'development_completion.json',{'experiment':'B01','source_commit':env['git_commit'],
        'aggregate_code_hash':env['code_hash'],'smoke_checkpoint':cp.relative_to(out).as_posix(),
        'straight_reference':next((out/'continuation/straight').rglob('step_000004.pt')).relative_to(out).as_posix(),
        'snapshot':snapshot.relative_to(out).as_posix(),'snapshot_hash':verify_snapshot(snapshot)['hash'],
        'continuation':result,'research_updates':0,'BPE_refitted':False})
    archive(out,'B01_development.zip')


def development_recovery(root,out):
    out.mkdir(parents=True,exist_ok=False);env,_=qualified(root,out)
    paths=list(Path('/kaggle/input').rglob('development_completion.json'));assert len(paths)==1
    original=paths[0].parent;completion=json.loads(paths[0].read_text(encoding='utf-8'))
    assert completion['experiment']=='B01' and completion['source_commit']==env['git_commit']
    manifest=verify_archive(original)
    snapshot=original/completion['snapshot'];assert verify_snapshot(snapshot)['hash']==completion['snapshot_hash']
    restored=restore_snapshot(snapshot,out/'restored');cp=restored/'checkpoints/step_000002.pt'
    receipt=roundtrip(cp,out/'roundtrip.pt','cuda')
    data=Path('/tmp/B01-data');prepare(root,data)
    payload,_=read_checkpoint(cp);settings=TrainSettings(**payload['metadata']['resolved']['training'])
    _,resumed,_=train(root,data,'B01',settings,4,out/'resumed',resume=cp)
    a,_=read_checkpoint(original/completion['straight_reference']);b,_=read_checkpoint(resumed)
    receipt.update(status='B01_published_development_fresh_CUDA_full_state_and_segmentation_passed',
        source_commit=env['git_commit'],aggregate_code_hash=env['code_hash'],environment=env,
        archive_sha256=manifest['archive_sha256'],verified_file_count=len(manifest['files']),
        continuation=compare_trajectory(a,b),research_updates=0,smoke_only_resumed_updates=2,
        feasibility=json.loads((original/'feasibility.json').read_text(encoding='utf-8')))
    atomic_json(out/'development_recovery.json',receipt)


def research(root,out):
    out.mkdir(parents=True,exist_ok=False);env,hardware=qualified(root,out);started=time.perf_counter()
    paths=list(Path('/kaggle/input').rglob('development_recovery.json'));assert len(paths)==1
    gate=json.loads(paths[0].read_text(encoding='utf-8'))
    assert gate['status']=='B01_published_development_fresh_CUDA_full_state_and_segmentation_passed'
    assert gate['source_commit']==env['git_commit'] and gate['aggregate_code_hash']==env['code_hash']
    assert all(gate['exact_state'].values()) and all(gate['continuation']['exact'].values())
    assert gate['additional_optimizer_updates']==gate['research_updates']==0
    assert gate['feasibility']['time_feasible'] and gate['feasibility']['storage_feasible']
    assert hardware['free_disk_bytes']>gate['feasibility']['storage_required_bytes']
    assert hardware['session_budget_seconds']>gate['feasibility']['projected_pipeline_with_margin_seconds']
    atomic_json(out/'development_recovery_gate.json',gate)
    data=Path('/tmp/B01-data');atomic_json(out/'dataset_validation.json',{'manifest':prepare(root,data),'BPE_refitted':False})
    atomic_json(out/'CUDA_architecture_audit.json',audit(root,data,'cuda'))
    settings=TrainSettings(device='cuda',run_kind='research',eval_every=250,checkpoint_every=500)
    atomic_json(out/'execution_plan.json',{'engine':'python -m nano_lm.train','settings':asdict(settings),
        'policy':STOP_POLICY,'policy_hash':digest(STOP_POLICY),'maximum':MAXIMUM,
        'initializations_authorized':1,'planned_segment_targets':list(range(250,5001,250)),
        'test_likelihood_evaluations_authorized':0})
    rule=StoppingRule();resume=None;segment_records=[];io=[]
    import nano_lm.train as engine
    import nano_lm.storage as storage
    original_save,original_export=engine.save_checkpoint,storage.export_run
    def measured(name,function):
        def call(*args,**kwargs):
            began=time.perf_counter();result=function(*args,**kwargs)
            io.append({'operation':name,'seconds':time.perf_counter()-began,'reference':str(args[0])})
            return result
        return call
    engine.save_checkpoint=measured('checkpoint',original_save)
    storage.export_run=measured('export',original_export)
    original_argv=sys.argv
    for target in range(250,MAXIMUM+1,250):
        assert rule.reason is None
        argv=['nano_lm.train','--root',str(root),'--dataset',str(data),'--experiment','B01','--steps',str(target),
            '--output',str(out/'runs'),'--export-root',str(out/'snapshots')]
        for key,value in asdict(settings).items():argv.extend(['--'+key.replace('_','-'),str(value)])
        if resume:argv.extend(['--resume',str(resume)])
        segment_start=time.perf_counter();sys.argv=argv
        try:engine.main()
        finally:sys.argv=original_argv
        runs=list((out/'runs').iterdir());assert len(runs)==1;run=runs[0]
        resume=run/f'checkpoints/step_{target:06d}.pt';payload,_=read_checkpoint(resume)
        state=payload['training_state'];row=state['history'][-1]
        assert payload['step']==len(state['history'])==target and state['training_tokens']==target*4096
        val=row['validation'];assert val['scored_bytes']==55514 and val['scored_tokens']==24386 and val['context_length']==128
        record=rule.observe(target,val['bpb'])
        atomic_json(out/f'stopping_decisions/step_{target:06d}.json',{**record,'checkpoint_sha256':file_hash(resume),
            'config_hash':payload['metadata']['config_hash'],'source_commit':env['git_commit'],
            'policy_hash':digest(STOP_POLICY),'tokens':state['training_tokens'],'source_bytes':state['training_bytes']})
        segment_records.append({'from':target-249,'through':target,'checkpoint_sha256':file_hash(resume),
            'wall_seconds':time.perf_counter()-segment_start,'planned_resume_from':target-250 if target>250 else None})
        print('B01 RESEARCH VALIDATION',json.dumps(record),flush=True)
        if rule.reason:break
    assert rule.reason and target<=MAXIMUM and len(rule.records)==target//250
    assert state['best_checkpoint']==f"step_{rule.best['step']:06d}.pt"
    stop={'experiment':'B01','run_id':run.name,'source_commit':env['git_commit'],'code_hash':env['code_hash'],
        'scientific_core_fingerprint':provenance(root)['scientific_core_fingerprint'],
        'config_hash':payload['metadata']['config_hash'],'maximum_authorized_updates':MAXIMUM,
        'actual_completed_updates':target,'full_validations':len(rule.records),'first_validation':250,
        'absolute_best':rule.best,'patience_history':rule.records,'policy':STOP_POLICY,'policy_hash':digest(STOP_POLICY),
        'stopping_step':target,'stopping_reason':rule.reason,'training_tokens':state['training_tokens'],
        'training_source_bytes':state['training_bytes'],'planned_segments':segment_records,
        'no_optimizer_updates_after_stopping_decision':True,'test_likelihood_evaluations':0}
    atomic_json(run/'stopping_summary.json',stop);atomic_json(out/'stopping_summary.json',stop)
    storage.export_run(run,out/'snapshots')
    engine.save_checkpoint=original_save;storage.export_run=original_export
    atomic_json(out/'io_timing.json',{'operations':io,'scope':'Original checkpoint/snapshot calls measured without changed arguments or optimization'})
    atomic_json(out/'completion.json',{**stop,'status':'B01_stopped_at_frozen_rule_or_budget',
        'parameters':PARAMETERS,'final_checkpoint':resume.relative_to(out).as_posix(),
        'best_checkpoint':(run/'checkpoints'/state['best_checkpoint']).relative_to(out).as_posix()})
    del payload;torch.cuda.empty_cache();analyze(root,run,out/'analysis','cuda')
    index=[{'snapshot':p.name,**verify_snapshot(p)} for p in (out/'snapshots').iterdir()]
    assert len(index)==2*len(rule.records)+1
    atomic_json(out/'snapshot_index.json',index)
    atomic_json(out/'wall_time.json',{'pipeline_seconds_before_archive':time.perf_counter()-started,
        'scope':'Research wrapper segments, stopping, generation and plots; excludes bootstrap/archive/publication/retrieval'})
    archive(out,'B01_seed42_final.zip',True)


def research_recovery(root,out):
    out.mkdir(parents=True,exist_ok=False);env,_=qualified(root,out)
    paths=list(Path('/kaggle/input').rglob('completion.json'));assert len(paths)==1
    original=paths[0].parent;completion=json.loads(paths[0].read_text(encoding='utf-8'))
    assert completion['status']=='B01_stopped_at_frozen_rule_or_budget' and completion['source_commit']==env['git_commit']
    manifest=verify_archive(original);target=completion['actual_completed_updates']
    snapshots=[(p,verify_snapshot(p)) for p in (original/'snapshots').iterdir()]
    assert len(snapshots)==2*completion['full_validations']+1
    snapshot,_=max((item for item in snapshots if item[1]['step']==target),key=lambda item:len(item[1]['files']))
    restored=restore_snapshot(snapshot,out/'restored');cp=restored/f'checkpoints/step_{target:06d}.pt'
    receipt=roundtrip(cp,out/'roundtrip.pt','cuda')
    payload,_=read_checkpoint(cp);assert replay(payload['training_state']['history']).records==completion['patience_history']
    from scripts.balanced.reproduce import reproduce
    reproduction=reproduce(root,original,out/'generation_reproduction','cuda')
    receipt.update(status='B01_published_research_fresh_CUDA_full_state_and_fixed_grid_passed',
        source_commit=env['git_commit'],environment=env,archive_sha256=manifest['archive_sha256'],
        verified_file_count=len(manifest['files']),verified_snapshot_count=len(snapshots),
        completed_updates=target,full_validations=completion['full_validations'],stopping_reason=completion['stopping_reason'],
        generation_reproduction=reproduction)
    atomic_json(out/'recovery.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['development','development-recovery','research','research-recovery'],required=True)
    a=p.parse_args()
    try:globals()[a.mode.replace('-','_')](a.root,a.output)
    except BaseException as exc:
        if a.output.exists():atomic_json(a.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'mode':a.mode})
        raise
