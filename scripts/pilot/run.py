"""Pre-flight and timing/preservation wrapper around the unchanged training CLI."""
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
from nano_lm.config import load_experiment
from nano_lm.model import GPT
from nano_lm.prepare import prepare, validate
from nano_lm.storage import export_run, verify_snapshot
from nano_lm.train import TrainSettings, environment, seed_all, train


def run(root, out, experiment='E01', development_only=False):
    if development_only and experiment!='E14':raise ValueError('Development publication is authorized only for E14')
    if experiment not in {'E01','E03','E04','E06','E08','E09','E10','E11','E13','E14'}:raise ValueError('Unsupported authorized experiment')
    total_start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=environment(root)
    assert not env['git_dirty'] and env['git_commit']
    assert env['code_hash']=='cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6','Core differs from audited E01'
    assert env['python']=='3.11.14' and env['torch']=='2.5.1+cu124' and env['cuda']=='12.4'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
    expected=(root/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines()
    normalize=lambda lines: sorted(x for x in lines if x and not x.startswith('-e '))
    assert normalize(env['packages'])==normalize(expected),'Qualified dependency freeze mismatch'
    atomic_json(out/'environment.json',env)
    hardware={'kernel_id':os.environ['NANO_KERNEL_ID'],'run_type':os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),'session_limit_seconds':int(os.environ['NANO_SESSION_LIMIT_SECONDS']),'gpu_count':torch.cuda.device_count(),'training_device':'cuda:0','devices':[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory,'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())],'disk_free_bytes':shutil.disk_usage(out).free,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    assert hardware['disk_free_bytes']>4*2**30 and hardware['session_limit_seconds']>=3600
    assert hardware['devices'][0]['free_bytes']>2*2**30
    atomic_json(out/'hardware.json',hardware)
    data=Path('/tmp/pilot-data');manifest=prepare(data,transport='http');validate(data)
    reference=json.loads((root/'docs/evidence/dataset_manifest.json').read_text())
    assert all(manifest[k]==reference[k] for k in ['source','split','partitions','tokenizers','manifest_hash'])
    atomic_json(out/'dataset_validation.json',{'all_reference_fields_match':True,'manifest':manifest})
    config,defaults,variant=load_experiment(root,experiment,65)
    positions={'E01':'sinusoidal','E03':'rope','E04':'learned','E06':'sinusoidal','E08':'sinusoidal','E09':'sinusoidal','E10':'sinusoidal','E11':'sinusoidal','E13':'sinusoidal','E14':'sinusoidal'}
    parameters=2808960 if experiment=='E14' else 1021824 if experiment=='E13' else 1213440 if experiment=='E10' else 3584256 if experiment=='E09' else 2694528 if experiment=='E08' else 1829376 if experiment=='E04' else 1804800
    expected_override = ({} if experiment=='E01' else {'d_model':240} if experiment=='E14' else {'d_model':144} if experiment=='E13' else {'context_length':64} if experiment=='E11' else {'mlp_expansion':2} if experiment=='E10' else {'n_heads':4} if experiment=='E06' else {'n_layers':8} if experiment=='E09' else {'n_layers':6} if experiment=='E08' else {'position':positions[experiment]})
    assert variant['model_overrides']==expected_override and variant['tokenizer']=='character'
    assert GPT(config).parameter_count()==parameters
    settings=TrainSettings(seed=42,device='cuda',precision='float32',run_kind='research',eval_every=250,checkpoint_every=500)
    settings.validate(config.context_length)
    from context_audit import token_budget
    budget_info=token_budget(config,settings)
    assert settings.tokens_per_update==4096 and settings.micro_batch_size==4
    expected_model={'vocab_size':65,**json.loads((root/'configs/baseline.json').read_text())['model']}
    expected_model['position']=positions[experiment]
    if experiment=='E06':expected_model['n_heads']=4
    if experiment=='E10':expected_model['mlp_expansion']=2
    if experiment=='E11':expected_model['context_length']=64
    if experiment in {'E13','E14'}:expected_model['d_model']=144 if experiment=='E13' else 240
    if experiment in {'E08','E09'}:expected_model['n_layers']=6 if experiment=='E08' else 8
    assert config.to_dict()==expected_model
    assert defaults['optimizer']=='AdamW' and defaults['learning_rate']==settings.learning_rate and defaults['scheduler']=='constant'
    # Separate one-update labeled smoke tests CUDA backward and checkpoint reload.
    smoke_settings=TrainSettings(device='cuda',run_kind='smoke',eval_every=1,checkpoint_every=1,eval_max_bytes=512)
    if experiment=='E03':
        from rope_audit import audit
        atomic_json(out/'rope_audit.json',audit(root,'cuda'))
    if experiment=='E04':
        from learned_audit import audit
        atomic_json(out/'learned_audit.json',audit(root,'cuda'))
    if experiment=='E06':
        from heads_audit import audit
        atomic_json(out/'heads_audit.json',audit(root,'cuda'))
    if experiment in {'E08','E09','E10','E11','E13','E14'}:
        if experiment in {'E13','E14'}:
            from width_audit import audit
            atomic_json(out/'width_audit.json',audit(root,'cuda',experiment))
        elif experiment=='E11':
            from context_audit import audit
            atomic_json(out/'context_audit.json',audit(root,'cuda'))
        elif experiment=='E10':
            from mlp_audit import audit
            atomic_json(out/'mlp_audit.json',audit(root,'cuda'))
        else:
            from depth_audit import audit
            atomic_json(out/'depth_audit.json',audit(root,'cuda',experiment))
        subprocess.run([sys.executable,'-m','pytest','-q','--basetemp',f'/tmp/{experiment.lower()}-pytest'],cwd=root,check=True)
    _,checkpoint,_=train(root,data,experiment,smoke_settings,1,out/'preflight_smoke')
    payload,_=read_checkpoint(checkpoint);model,_=load_for_inference(checkpoint)
    assert payload['step']==1 and model.parameter_count()==parameters
    verify_snapshot(export_run(checkpoint.parent.parent,out/'preflight_snapshots'))
    if experiment in {'E08','E09','E10','E11','E13','E14'}:
        from nano_lm.storage import restore_snapshot
        from nano_lm.verification import equal_nested
        snapshot=next((out/'preflight_snapshots').iterdir())
        restored=restore_snapshot(snapshot,out/'preflight_snapshots'/'restored')
        recovered,_=read_checkpoint(restored/'checkpoints'/checkpoint.name)
        assert all(equal_nested(payload[k],recovered[k]) for k in ['model','optimizer','scheduler','scaler','rng','training_state'])
        budget=100*checkpoint.stat().st_size+2*2**30
        assert shutil.disk_usage(out).free>budget,'Insufficient checkpoint/snapshot/archive headroom'
        archive=out/'preflight_snapshots'/'smoke.zip'
        with zipfile.ZipFile(archive,'x') as z:z.write(checkpoint,'checkpoint.pt')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            import hashlib
            assert hashlib.sha256(z.read('checkpoint.pt')).hexdigest()==file_hash(checkpoint)
        atomic_json(out/'storage_preflight.json',{'status':'smoke_export_restore_and_ZIP_verified','smoke_checkpoint_bytes':checkpoint.stat().st_size,'smoke_checkpoint_SHA256':file_hash(checkpoint),'smoke_archive_SHA256':file_hash(archive),'ZIP_CRC_safe_paths_and_embedded_checkpoint_hash_passed':True,'conservative_required_free_bytes':budget,'available_bytes':shutil.disk_usage(out).free,'plan':'20 checkpoints plus21 best/latest snapshots and archive;100 weight-file equivalents plus2GiB margin'})
    if experiment in {'E11','E13','E14'}:
        from nano_lm.bpe import from_artifact
        from nano_lm.evaluate import evaluate
        full_validation=evaluate(model.to('cuda'),from_artifact(payload['tokenizer']), (data/'validation.txt').read_text(encoding='utf-8'))
        assert full_validation['scored_bytes']==55514 and full_validation['context_length']==config.context_length
        atomic_json(out/'full_validation_preflight.json',full_validation)
    if experiment in {'E13','E14'}:
        from preflight_restore import audit_restore
        atomic_json(out/'CUDA_smoke_state_restore.json',audit_restore(checkpoint,out/'preflight_snapshots'/'state_roundtrip.pt'))
    del model,payload;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(checkpoint.parent.parent),'--output',str(out/'preflight_analysis'),'--device','cuda','--development-check'],check=True)
    atomic_json(out/'preflight.json',{'status':'passed','parameters':parameters,'model':config.to_dict(),'settings':asdict(settings),**budget_info,'updates_authorized':5000,'no_test_evaluation':True,'storage_plan':'private saved notebook output version; verify and retrieve archive after automatic publication; restore from verified snapshot if interrupted','prior_saved_output_recovery':'docs/qualifications/evidence/rayyan_fresh_gpu_recovery.json'})
    if development_only:
        from development import publish_development
        publish_development(root,out,data,experiment,checkpoint,env)
        return
    if experiment=='E14':
        from development import require_recovery_receipt
        require_recovery_receipt(root,out)
    # Timing hooks forward calls unchanged. No optimizer/model/evaluation changes.
    import nano_lm.train as engine
    import nano_lm.storage as storage
    timers=[];journal_seconds=[0.]
    def timed(name, function):
        def call(*args,**kwargs):
            start=time.perf_counter();result=function(*args,**kwargs)
            timers.append({'operation':name,'seconds':time.perf_counter()-start,'reference':str(result if name=='export' else args[0])})
            return result
        return call
    engine.save_checkpoint=timed('checkpoint',engine.save_checkpoint)
    storage.export_run=timed('export',storage.export_run)
    original_event=engine.append_event
    def event(path,value):
        start=time.perf_counter();original_event(path,value);journal_seconds[0]+=time.perf_counter()-start
        if value['event']=='step' and value['metric']['step']%250==0:
            print('PILOT PROGRESS',json.dumps(value['metric']),flush=True)
    engine.append_event=event
    argv=[sys.executable,'--root',str(root),'--dataset',str(data),'--experiment',experiment,'--seed','42','--steps','5000','--device','cuda','--precision','float32','--run-kind','research','--tokens-per-update','4096','--micro-batch-size','4','--optimizer','AdamW','--learning-rate','0.0003','--weight-decay','0.01','--gradient-clip','1.0','--eval-every','250','--checkpoint-every','500','--conditioning-bytes','256','--eval-max-bytes','0','--eval-mode','sliding','--eval-batch-size','32','--output',str(out/'runs'),'--export-root',str(out/'snapshots')]
    atomic_json(out/'execution_command.json',{'entrypoint':'python -m nano_lm.train','arguments':argv[1:],'timing_hooks':'checkpoint/export/journal calls only; original arguments and return values preserved'})
    sys.argv=argv;began=time.perf_counter()
    engine.main()
    cli_seconds=time.perf_counter()-began
    atomic_json(out/'io_timing.json',{'records':timers,'journal_seconds':journal_seconds[0],'cli_wall_seconds':cli_seconds,'scope':'actual original checkpoint/export functions plus journal fsync; excludes notebook publication and network retrieval'})
    runs=list((out/'runs').iterdir());assert len(runs)==1
    directory=runs[0];final=directory/'checkpoints/step_005000.pt';payload,_=read_checkpoint(final)
    state=payload['training_state'];history=state['history'];val=[r for r in history if 'validation' in r]
    assert payload['step']==5000 and len(history)==5000 and [r['step'] for r in history]==list(range(1,5001))
    assert [r['step'] for r in val]==list(range(250,5001,250))
    assert state['training_tokens']==20480000 and state['training_bytes']==20480000 and state['amp_skipped_updates']==0
    assert all(r['lr']==0.0003 and r['accumulation']==budget_info['accumulation'] for r in history)
    assert all(r['validation']['scored_bytes']==55514 and r['validation']['mode']=='sliding' and r['validation']['context_length']==config.context_length and r['validation']['conditioning_bytes']==256 for r in val)
    for step in [1000,2000,3000,4000,5000]:assert (directory/f'checkpoints/step_{step:06d}.pt').exists()
    assert (directory/'checkpoints'/state['best_checkpoint']).exists()
    atomic_json(out/'completion.json',{'status':'5000_completed_optimizer_updates','run_id':directory.name,'source_commit':env['git_commit'],'code_hash':env['code_hash'],'final_checkpoint':str(final.relative_to(out)),'best_checkpoint':str((directory/'checkpoints'/state['best_checkpoint']).relative_to(out)),'validation_steps':[r['step'] for r in val],'wrapper_seconds_before_postprocessing':time.perf_counter()-total_start})
    # Committed analysis script reads only genuine committed checkpoint history.
    subprocess.run([sys.executable,str(root/'scripts/pilot/analyze.py'),'--root',str(root),'--run',str(directory),'--output',str(out/'analysis'),'--device','cuda'],check=True)
    snapshot_index=[]
    for snapshot in (out/'snapshots').iterdir():
        m=verify_snapshot(snapshot);snapshot_index.append({'snapshot':snapshot.name,'step':m['step'],'hash':m['hash'],'files':m['files']})
    atomic_json(out/'snapshot_index.json',snapshot_index)
    atomic_json(out/'wall_time.json',{'wrapper_total_seconds_before_archive':time.perf_counter()-total_start,'cli_wall_seconds':cli_seconds,'note':'excludes environment bootstrap, archive construction, platform publication and download'})
    # Preserve every periodic/validation checkpoint once plus all logs and manifests.
    archive=out.parent/f'{experiment}_seed42_pilot.zip';began=time.perf_counter()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in out.rglob('*'):
            if path.is_file() and 'snapshots' not in path.relative_to(out).parts and 'preflight_smoke' not in path.relative_to(out).parts and 'preflight_analysis' not in path.relative_to(out).parts and 'preflight_snapshots' not in path.relative_to(out).parts:
                z.write(path,path.relative_to(out).as_posix())
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and 'snapshots' not in p.relative_to(out).parts and 'preflight_smoke' not in p.relative_to(out).parts and 'preflight_analysis' not in p.relative_to(out).parts and 'preflight_snapshots' not in p.relative_to(out).parts}
    atomic_json(out.parent/'artifact_manifest.json',{'source_commit':env['git_commit'],'run_id':directory.name,'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,'files':files,'archive_seconds':time.perf_counter()-began,'snapshot_policy':'all immutable snapshots remain in saved output; archive contains complete run with all checkpoints, analysis and supplemental evidence'})
    print('PILOT COMPLETE ARCHIVE',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--development-only',action='store_true');p.add_argument('--experiment',choices=['E01','E03','E04','E06','E08','E09','E10','E11','E13','E14'],default='E01')
    a=p.parse_args()
    try:run(a.root,a.output,a.experiment,a.development_only)
    except BaseException as e:
        if a.output.exists():
            (a.output/'failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)},indent=2))
        raise
