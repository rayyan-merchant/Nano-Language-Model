"""Generate E12 reporting from checksum-verified artifacts and reviewed references."""
import argparse,csv,json,math,shutil,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nano_lm.checkpoint import read_checkpoint
from nano_lm.artifacts import file_hash
from scripts.pilot.sampling_audit import replay


def report(root,artifacts,verification,recovery,output,kernel,recovery_kernel):
    read=lambda p:json.loads(p.read_text())
    s=read(artifacts/'analysis/summary.json');done=read(artifacts/'completion.json')
    verified=read(verification);cuda=read(recovery)
    assert s['completed_optimizer_updates']==5000 and s['parameters']==1804800
    assert verified['status']=='independent_download_reconstruction_all_hashes_and_CPU_inference_passed'
    assert cuda['status']=='fresh_saved_output_CUDA_state_restore_passed_no_updates' and cuda['additional_optimizer_updates']==0
    assert all(cuda['exact_state'].values())
    cp=artifacts/done['final_checkpoint'];p,_=read_checkpoint(cp)
    assert p['config']['context_length']==256 and p['config']['n_heads']==6
    sampling=replay(p);history=p['training_state']['history'];val=[r for r in history if 'validation' in r]
    assert sampling['accumulation']==4 and len(val)==20
    assert all(r['validation']['scored_bytes']==55514 and r['validation']['context_length']==256 for r in val)
    env=read(artifacts/'environment.json');audit=read(artifacts/'context_audit.json')
    assert audit['status']=='passed' and audit['device']=='cuda'
    output.mkdir(parents=True,exist_ok=False);(output/'evidence').mkdir()
    for file in (artifacts/'analysis').iterdir():
        if file.is_dir():shutil.copytree(file,output/file.name)
        elif file.is_file():shutil.copyfile(file,output/file.name)
    for file in artifacts.glob('*.json'):shutil.copyfile(file,output/'evidence'/file.name)
    shutil.copyfile(verification,output/'evidence/independent_verification.json');shutil.copyfile(recovery,output/'evidence/fresh_final_CUDA_recovery.json')
    (output/'evidence/sampling_audit.json').write_text(json.dumps(sampling,indent=2))
    references={'E01':root/'report/pilots/E01_seed42','E11':root/'report/pilots/E11_seed42'}
    summaries={eid:read(folder/'summary.json') for eid,folder in references.items()};summaries['E12']=s
    reference_validation={}
    for eid,folder in references.items():
        reference_env=read(folder/'evidence/environment.json')
        assert all(reference_env[k]==env[k] for k in ['code_hash','python','torch','cuda','gpu'])
        normalize=lambda e:sorted(x for x in e['packages'] if not x.startswith('-e '))
        assert normalize(reference_env)==normalize(env)
        dataset_path=folder/'evidence/dataset_validation.json'
        data=read(dataset_path)['manifest']
        assert data==read(artifacts/'dataset_verification.json')['manifest']
        reference_validation[eid]={'same_scientific_profile':True,'same_dataset_manifest':True,'environment_file_sha256':file_hash(folder/'evidence/environment.json'),'dataset_receipt_file_sha256':file_hash(dataset_path)}
    values={eid:list(csv.DictReader((folder/'validation.csv').open())) for eid,folder in references.items()}
    values['E12']=[{'step':r['step'],**r['validation']} for r in val]
    for eid in values:
        assert [int(r['step']) for r in values[eid]]==list(range(250,5001,250))
        assert summaries[eid]['code_hash']==s['code_hash'] and summaries[eid]['parameters']==1804800
    rows=[]
    for i in range(20):
        row={'step':250*(i+1),'E12_training_loss':val[i]['training_loss'],'E12_gradient_norm':val[i]['gradient_norm'],'E12_training_seconds':val[i]['train_seconds']}
        for eid in values:
            for metric in ['loss','bpb','perplexity']:row[eid+'_'+metric]=float(values[eid][i][metric])
        row['delta_E12_minus_E01_bpb']=row['E12_bpb']-row['E01_bpb'];rows.append(row)
    comparison={'reference_scope':'Reviewed Git report values; E12 independently recovered here. E01/E11 original archive/recovery receipts remain in their reviewed report directories. No fresh E01/E11 inference claimed.','reference_file_sha256':{eid:{name:file_hash(folder/name) for name in ['summary.json','validation.csv','generation.json']} for eid,folder in references.items()},'reference_profile_checks':reference_validation,'summaries':summaries,'matched_validation':rows,'E12_sampling':sampling,'context_lengths':{'E11':64,'E01':128,'E12':256},'single_seed_only':True}
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    with (output/'matched_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for metric in ['loss','bpb']:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for eid,L in [('E11',64),('E01',128),('E12',256)]:ax.plot([r['step'] for r in rows],[r[eid+'_'+metric] for r in rows],marker='.',label=f'{eid}: {L} tokens')
        ax.set(xlabel='Completed optimizer updates',ylabel=metric,title='Measured seed42 context comparison');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'figures'/('context_'+metric+'.png'),dpi=200);fig.savefig(output/'figures'/('context_'+metric+'.pdf'));plt.close(fig)
    norms=[r['gradient_norm'] for r in history];io=read(artifacts/'io_timing.json');generation=read(output/'generation.json')
    assert len(generation['samples'])==100 and generation['protocol']==read(root/'configs/generation_protocol.json')
    quality=[]
    for eid,path in [('E01',references['E01']/'generation.json'),('E11',references['E11']/'generation.json'),('E12',output/'generation.json')]:
        grid=read(path);assert len(grid['samples'])==100 and grid['protocol']==read(root/'configs/generation_protocol.json')
        assert grid['protocol_source_sha256']==file_hash(root/'configs/generation_protocol.json')
        for sample in grid['samples']:
            text=sample['text'][len(sample['prompt']):];windows=[text[i:i+8] for i in range(len(text)-7)]
            quality.append({'experiment':eid,'policy':sample['checkpoint_policy'],'prompt':sample['prompt'],'temperature':sample['temperature'],'greedy':sample['greedy'],'seed':sample['seed'],'repeated_8gram_fraction':1-len(set(windows))/len(windows)})
    (output/'generation_quality.json').write_text(json.dumps(quality,indent=2))
    events=[json.loads(line) for line in (cp.parent.parent/'events.jsonl').read_text().splitlines()]
    attempts=sum(e['event']=='step' for e in events);assert attempts>=5000
    failures=[e for e in events if e['event'] in ['failure','resume']]
    text=f'''# E12 seed42 — context length 256

## Scope and research question

How does increasing causal context128→256 affect next-character prediction,
optimization, efficiency, memory and generated behavior? Longer conditioning
could support dependencies but also adds attention cost and changes training
window grouping. Improvements and regressions were both possible.
One authorized seed42 run; exactly5000 authoritative updates. No additional
research seeds, experiments, hybrid configurations or extensions were executed.
All conclusions are single-seed descriptive observations.

## Architecture and numerical audit

Only context_length differs from E01:256. Vocabulary65,width192,four blocks,
six heads/head32, original MLP192→768→192,GELU,sinusoidal positions,dropout0.1.
Trainable parameters1,804,800. CPU and actual qualified CUDA audits passed:
256×256 causal masks, positions0–255, explicit Q/K/V/sqrt32 reference,
future-token invariance, finite shifted-target CE/backward/all-block gradients,
train/eval dropout, overlength rejection, generation last256-token cropping,
and independent sequential sliding evaluation. Initial trainable tensors match
E01 under seed42; sinusoidal buffers do not consume initialization RNG.
Synthetic audit fixtures are development evidence, never research observations.
Scientific algorithms, tokenizer, training engine and generation protocol unchanged.

## Source, environment, data and optimizer

Clean trained SHA:`{s['source_commit']}`; code hash:`{s['code_hash']}`;
run:`{s['run_id']}`; config hash:`{s['config_hash']}`.
[Research saved notebook](https://www.kaggle.com/code/{kernel}) and
[fresh final recovery](https://www.kaggle.com/code/{recovery_kernel}).
TeslaT4 CUDA0; isolated Python{env['python']},torch{env['torch']},CUDA{env['cuda']},
NumPy1.26.4,tokenizers0.21.0, exact qualified46-package freeze.
Raw dataset SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed;
character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
Partitions1,003,854/55,770/55,770 bytes, unchanged boundaries and sampling policy.
Fresh initialization, AdamW3e-4 constant, weight decay0.01,clip1,float32CUDA,
no AMP,early stopping or test likelihood. Full state/RNG/private sampling retained.

## Context, token budget and validation policy

| Experiment | Context | Microbatch | Accumulation | Windows/update | Targets/update |
|---|---:|---:|---:|---:|---:|
| E11 |64|4|16|64|4096|
| E01 |128|4|8|32|4096|
| E12 |256|4|4|16|4096|

All5k budgets predict20,480,000 targets, but E12 samples80,000 windows versus
E01's160,000 and E11's320,000. Attention matrices are larger; equal targets do
not imply equalFLOPs,runtime,input traffic or identical source coverage.
Actual E12 repeated source-target bytes:{s['training_source_bytes']:,}; unique
byte coverage:{s['unique_target_bytes']:,}. Private sampler/coverage replay passed.
Full validation scores the same55,514 original suffix bytes once after256
conditioning-only bytes, every250 updates, dropoutoff,batch32. E12 uses256
preceding characters versus128/64 for E01/E11. Common prefix exclusion and
model context are conceptually distinct. BPB=NLL/(ln2×scoredbytes);perplexity=exp(loss).
Local positional indices reset in each canonical crop. No subsampling or test selection.

## Measurements and resource costs

Best update{s['best_validation']['step']}:loss{s['best_validation']['loss']:.8f},
BPB{s['best_validation']['bpb']:.8f}; final5000:loss{s['final_validation']['loss']:.8f},
BPB{s['final_validation']['bpb']:.8f},perplexity{s['final_validation']['perplexity']:.8f}.
Training{s['training_seconds']:.3f}s; full validations{s['validation_seconds']:.3f}s;
throughput{5000/s['training_seconds']:.5f}steps/s and{s['training_tokens']/s['training_seconds']:.3f}targets/s;
peak allocated{s['peak_gpu_bytes']/2**20:.3f}MiB. Allocation excludes reserved/driver memory.
First train loss{s['initial_recorded_training']['loss']:.8f}; final train loss{s['final_training_loss']:.8f};
last100mean{s['final_training_trailing_100_loss']:.8f}. Gradient norm range
{min(norms):.6f}–{max(norms):.6f}; mean{statistics.mean(norms):.6f};
{sum(n>1 for n in norms)}/5000 updates exceed clipping threshold.
Per-step losses,norms,LR,exposure,timing and accumulation are in measured_history.json.
Training times exclude validation/checkpoint/export. Actual engine wall
{io['engine_wall_seconds']:.3f}s; checkpoint{sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'):.3f}s;
export{sum(r['seconds'] for r in io['records'] if r['operation']=='export'):.3f}s;
journal{io['journal_seconds']:.3f}s. Bootstrap,publication,download and fresh recovery
are separate stages; no unmeasured total time is invented.

## All twenty matched validation points

| Update | E01 loss | E12 loss | E01 BPB | E12 BPB | E11 BPB | E12 train loss |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in rows:text+=f"|{r['step']}|{r['E01_loss']:.8f}|{r['E12_loss']:.8f}|{r['E01_bpb']:.8f}|{r['E12_bpb']:.8f}|{r['E11_bpb']:.8f}|{r['E12_training_loss']:.8f}|\n"
    text+='\n## E01/E12 and genuine E11 context framework\n\n| Metric | E11 | E01 | E12 |\n|---|---:|---:|---:|\n'
    for title,key,sub,scale in [('BestBPB','best_validation','bpb',1),('FinalBPB','final_validation','bpb',1),('Trainingseconds','training_seconds',None,1),('Validationseconds','validation_seconds',None,1),('AllocatedMiB','peak_gpu_bytes',None,2**20)]:
        vals=[(summaries[e][key][sub] if sub else summaries[e][key])/scale for e in ['E11','E01','E12']];text+=f'|{title}|'+ '|'.join(f'{v:.6f}' for v in vals)+'|\n'
    delta=s['best_validation']['bpb']-summaries['E01']['best_validation']['bpb']
    text+=f'\nObserved bestBPB deltaE12−E01:{delta:+.8f}. These are matching recorded targets under different conditioning limits and gradient-window grouping. SeparateT4 sessions are a timing confound. No single-seed statistical significance,causal coherence claim or future convergence guarantee. E11 measurements come from its reviewed independently recovered report; E11 was not launched here.\n'
    for fig in sorted((output/'figures').glob('*.png')):text+=f'\n![Measured {fig.stem}](figures/{fig.name})\n'
    text+=f'''\n## Fixed genuine generation and qualitative assessment

[All100 unedited outputs](generation.md) and generation.json retain both policies,
five predetermined prompts,greedy and temperatures0.7/1/1.2,seeds42/123/456,
256new tokens. Cropping uses last256 tokens. Best/final duplicate outputs are
explicitly verified when they share weights:{verified['best_equals_final']}.
All300 E11/E01/E12 descriptive8-gram repetition rows are in generation_quality.json;
this is not an automatic grammar,coherence or memorization score.
See interpretation.md for evidence-based speaker-label,grammar,spelling,prompt
responsiveness and loop observations from the full genuine outputs.

## Integrity,persistence,failures and recovery

Archive{verified['archive_bytes']:,}bytes,SHA256`{verified['archive_sha256']}`.
ZIP CRC,safe unique paths and every{verified['archive_entries']}filehash passed.
20full checkpoint states/21 reconstructed immutable snapshots verified,including
all optimizer/scheduler counters,5000history rows and20validations.
Best/finalCPU inference passed; fresh compatibleCUDA restoration preserved full
model/optimizer/scheduler/scaler/RNG/sampling/history at5000→5000 with0additional
research updates. Labeled smoke persisted before substantive research and exact
continued smoke state was verified. All64MiB-or-smaller transfer parts checked.
Working-directory files alone were never treated as durable publication.
Known journal step attempts:{attempts}; failure/resume events:{len(failures)}.
Any lost/replayed attempts must be recorded separately from authoritative history;
this original journal count alone cannot reconstruct unknown volatile execution.
Development smoke/benchmark updates are separately labeled,not research results.
Large checkpoints,datasets and archives stay outsideGit.

## Limitations and implications

One dataset/seed; noSD or significance estimate. Longer context changes available
conditioning,attention cost,window sampling/dropout draws and optimization grouping.
Training-loss/validation-loss gaps compare dropout-on sampled windows against
dropout-off full sliding validation and do not prove memorization or overfitting.
Later optimized architecture selection should weigh actualBPB,observedtext quality,
memory and measured total cost; combining favorable factors or adding seeds requires
separate approval. Preserve negative outcomes and do not extend beyond5000.
'''
    (output/'E12_seed42_analysis.md').write_text(text)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','artifacts','verification','recovery','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kernel',required=True);parser.add_argument('--recovery-kernel',required=True)
    a=parser.parse_args();report(a.root,a.artifacts,a.verification,a.recovery,a.output,a.kernel,a.recovery_kernel)
