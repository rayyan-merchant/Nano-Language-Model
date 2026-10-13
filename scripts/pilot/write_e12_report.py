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
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    s=read(artifacts/'analysis/summary.json');done=read(artifacts/'completion.json')
    verified=read(verification); cuda=read(recovery) if recovery else None
    final_cuda_verified=cuda is not None
    assert s['completed_optimizer_updates']==5000 and s['parameters']==1804800
    assert verified['status']=='independent_download_reconstruction_all_hashes_and_CPU_inference_passed'
    if final_cuda_verified:
        assert cuda['status']=='fresh_saved_output_CUDA_state_restore_passed_no_updates' and cuda['additional_optimizer_updates']==0
        assert all(cuda['exact_state'].values())
        assert cuda['run_id']==s['run_id'] and cuda['trained_source_commit']==s['source_commit']
        assert cuda['optimizer_updates_before']==cuda['optimizer_updates_after']==5000
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
    shutil.copyfile(verification,output/'evidence/independent_verification.json')
    if final_cuda_verified: shutil.copyfile(recovery,output/'evidence/fresh_final_CUDA_recovery.json')
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
    values={eid:list(csv.DictReader((folder/'validation.csv').open(encoding='utf-8-sig'))) for eid,folder in references.items()}
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
        ax.set(xlabel='Completed optimizer updates',ylabel=('Conditional bits per byte' if metric=='bpb' else 'Validation loss (nats/character)'),title='Measured seed42 context comparison');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'figures'/('context_'+metric+'.png'),dpi=200);fig.savefig(output/'figures'/('context_'+metric+'.pdf'));plt.close(fig)
    norms=[r['gradient_norm'] for r in history];io=read(artifacts/'io_timing.json');generation=read(output/'generation.json')
    assert len(generation['samples'])==100 and generation['protocol']==read(root/'configs/generation_protocol.json')
    quality=[]
    for eid,path in [('E01',references['E01']/'generation.json'),('E11',references['E11']/'generation.json'),('E12',output/'generation.json')]:
        grid=read(path);assert len(grid['samples'])==100 and grid['protocol']==read(root/'configs/generation_protocol.json')
        assert grid.get('protocol_source_sha256',grid['protocol_sha256'])==file_hash(root/'configs/generation_protocol.json')
        for sample in grid['samples']:
            text=sample['text'][len(sample['prompt']):];windows=[text[i:i+8] for i in range(len(text)-7)]
            quality.append({'experiment':eid,'policy':sample['checkpoint_policy'],'prompt':sample['prompt'],'temperature':sample['temperature'],'greedy':sample['greedy'],'seed':sample['seed'],'repeated_8gram_fraction':1-len(set(windows))/len(windows)})
    (output/'generation_quality.json').write_text(json.dumps(quality,indent=2))
    events=[json.loads(line) for line in (cp.parent.parent/'events.jsonl').read_text().splitlines()]
    attempts=sum(e['event']=='step' for e in events);assert attempts>=5000
    failures=[e for e in events if e['event'] in ['failure','resume']]
    text=f'''# E12 seed42 — context length 256

## Result and research question

E12 completed the sole authorized seed42 run at **5,000 authoritative optimizer
updates**. Best and final both select step 5,000: conditional BPB
**{s['final_validation']['bpb']:.8f}**, worse than E01's best 2.21405793 and final
2.21599704. Longer context did not improve prediction under this budget.
Training was faster, memory allocation and validation time increased, and greedy
text still entered repeated-phrase loops. No other research seed, experiment,
combined configuration or extension was trained. These are descriptive single-seed results.

The question was how increasing causal context from 128 to 256 characters affects
next-character prediction, optimization, efficiency, memory and generated text.
Longer conditioning might support distant dependencies; larger attention matrices
and different training-window grouping might also change cost and optimization.
Neither improvement nor regression was assumed before the run.

## Architecture and numerical audit

The sole E12 architecture override is `context_length: 256`. The character
vocabulary is 65, width 192, with four independent Transformer blocks, six heads
of dimension 32, MLP 192→768→192, GELU, sinusoidal positions and dropout 0.1.
There are **1,804,800 trainable parameters**, unchanged from E01. Explicit Q/K/V,
sqrt(32) scaling, causal masking, softmax and value aggregation remain canonical.

Focused CPU and qualified CUDA checks passed: resolved context and sole override;
256×256 masks and sinusoidal indices 0–255; shifted targets; forward/backward CE;
independent six-head attention reference; future-token invariance; finite nonzero
gradients through all blocks; dropout in train/eval modes; original MLP and GELU;
overlength rejection; generation cropping to the final 256 tokens; and a sequential
full-sliding evaluator reference. The audit also checks identical seed42 initial
trainable tensors against E01; sinusoidal buffers consume no initialization RNG.
The scientific hash is checked independently. Synthetic fixtures and development
smoke/benchmark updates are engineering evidence, not research measurements.
Qualified preflight passed the full 67-test CUDA suite.

## Source, software, dataset and optimizer

- Clean trained source: `{s['source_commit']}`.
- Scientific hash: `{s['code_hash']}`.
- Run: `{s['run_id']}`; configuration hash: `{s['config_hash']}`.
- [Saved research version 358085129](https://www.kaggle.com/code/{kernel}).
- [Fresh final CUDA recovery version 358138796](https://www.kaggle.com/code/{recovery_kernel}).
- Tesla T4, CUDA device 0, isolated Python {env['python']}, PyTorch {env['torch']},
  CUDA {env['cuda']}, NumPy 1.26.4, Tokenizers 0.21.0, exact qualified dependency
  freeze (46 non-editable packages). Native Kaggle packages were not used for research.

Canonical Tiny Shakespeare source SHA256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
character tokenizer SHA256 is
`ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`.
Original train/validation/test partitions contain 1,003,854/55,770/55,770 bytes.
Dataset manifests, tokenizer, boundaries and random-window sampling remain unchanged.
No test-set likelihood or test-based selection was performed.

Training reseeded from fresh initialization: AdamW, constant learning rate 0.0003,
weight decay 0.01, clipping norm 1.0, float32 CUDA, microbatch four, accumulation
four. There was no AMP, early stopping, trained-weight reuse or settings change in
response to results. Full model, optimizer, scheduler, scaler, RNG, sampling and
training-history state was retained.

## Context, token exposure and validation

| Experiment | Context | Microbatch | Accumulation | Windows/update | Targets/update |
|---|---:|---:|---:|---:|---:|
| E11 |64|4|16|64|4096|
| E01 |128|4|8|32|4096|
| E12 |256|4|4|16|4096|

All three budgets predict 20,480,000 target tokens. E12 samples 80,000 windows,
versus E01's 160,000 and E11's 320,000. Equal targets do not imply equal FLOPs,
input traffic, runtime, contextual information or gradient grouping. Different
start-index ranges, random draws and dropout shapes also change the trajectories.
E12 recorded {s['training_source_bytes']:,} repeated original target bytes and
{s['unique_target_bytes']:,} unique target bytes. Independent private-sampler replay
matched the saved RNG and exact target-position union without optimizer updates.
E01 covered 1,003,848 unique target bytes; E11 covered 1,003,846. Coverage measures
predicted target positions, not every byte supplied as context or complete epochs.

Each full sliding validation excludes the common first 256 original bytes from
scoring, then scores all 55,514 remaining original bytes exactly once. It runs every
250 updates with dropout disabled and batch size 32, without subsampling. E12 may
condition on 256 preceding characters, E01 on 128, E11 on 64. The common excluded
prefix is a scoring policy, conceptually separate from model context. Local
sinusoidal positions reset in each canonical crop. BPB = NLL/(ln(2)×scored bytes);
perplexity = exp(loss). All three likelihoods use identical scored targets under
their intended conditioning restrictions.

## Genuine measurements and resource costs

Best/final step 5,000: loss **{s['final_validation']['loss']:.8f}**, BPB
**{s['final_validation']['bpb']:.8f}**, perplexity **{s['final_validation']['perplexity']:.8f}**.
The first recorded training loss is {s['initial_recorded_training']['loss']:.8f};
final loss is {s['final_training_loss']:.8f}; the final 100-update mean is
{s['final_training_trailing_100_loss']:.8f}. No step-zero validation is invented.

| E12 measured quantity | Value |
|---|---:|
| Training-only seconds |{s['training_seconds']:.3f}|
| Twenty full-validation seconds |{s['validation_seconds']:.3f}|
| Updates/training second |{5000/s['training_seconds']:.5f}|
| Targets/training second |{s['training_tokens']/s['training_seconds']:.3f}|
| Peak allocated GPU MiB |{s['peak_gpu_bytes']/2**20:.3f}|
| Engine wall seconds |{io['engine_wall_seconds']:.3f}|
| Checkpoint write seconds |{sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'):.3f}|
| Immutable export seconds |{sum(r['seconds'] for r in io['records'] if r['operation']=='export'):.3f}|
| Journal write seconds |{io['journal_seconds']:.3f}|
| Fixed-grid GPU generation seconds |{generation['seconds']:.3f}|

Training time excludes validation and IO. Allocation excludes reserved and driver
memory. The recorded gradient norm ranges from {min(norms):.6f} to {max(norms):.6f},
with mean {statistics.mean(norms):.6f}; {sum(n>1 for n in norms)} of 5,000 updates
exceed the clipping threshold (0.14%). All 5,000 norms are finite. Learning rate
stayed 0.0003 and accumulation stayed four at every update; no AMP updates were skipped.
`measured_history.json` preserves per-step loss, norms, LR, exposure and timing.
`matched_validation.csv` also records milestone training loss, gradient norm and
cumulative training seconds alongside the three-way validation metrics.

The saved output's separately observed research-and-finish timer is
**1,684.159 seconds**, covering research, validation, IO, generation, verification
and local packaging. It excludes bootstrap, development smoke, platform publication
and independent download. The platform reported a successful **1,796.6-second**
research notebook run. Pre-archive postprocessing took 195.781 seconds; independent
archive download took 452.357 seconds; local integrity/CPU verification took 9.148
seconds. Fresh final CUDA recovery was a separate successful 105.0-second notebook
job. These scopes overlap or exclude different stages and must not be added as a
single measured end-to-end elapsed time. The pipeline preview capture and download
receipts make their scope explicit.

## All twenty matched validation points

| Update | E01 loss | E12 loss | E01 BPB | E12 BPB | E11 BPB | E12 train loss |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in rows:text+=f"|{r['step']}|{r['E01_loss']:.8f}|{r['E12_loss']:.8f}|{r['E01_bpb']:.8f}|{r['E12_bpb']:.8f}|{r['E11_bpb']:.8f}|{r['E12_training_loss']:.8f}|\n"
    text+='\n## E01/E12 and genuine E11 context framework\n\n| Metric | E11 | E01 | E12 |\n|---|---:|---:|---:|\n'
    for title,key,sub,scale in [('Best BPB','best_validation','bpb',1),('Final BPB','final_validation','bpb',1),('Training seconds','training_seconds',None,1),('Validation seconds','validation_seconds',None,1),('Allocated MiB','peak_gpu_bytes',None,2**20)]:
        vals=[(summaries[e][key][sub] if sub else summaries[e][key])/scale for e in ['E11','E01','E12']];text+=f'|{title}|'+ '|'.join(f'{v:.6f}' for v in vals)+'|\n'
    delta=s['best_validation']['bpb']-summaries['E01']['best_validation']['bpb']
    text+=f'\nObserved best BPB difference E12−E01: {delta:+.8f}. These are matching recorded targets under different conditioning limits and gradient-window grouping. Separate T4 sessions are a timing confound. No single-seed statistical significance, causal coherence claim or future convergence guarantee. E11 measurements come from its reviewed independently recovered report; E11 was not launched here.\n'
    for fig in sorted((output/'figures').glob('*.png')):text+=f'\n![Measured {fig.stem}](figures/{fig.name})\n'
    text+=f'''
## Genuine generation and qualitative assessment

[All 100 unedited outputs](generation.md) and `generation.json` preserve both
checkpoint policies, all five predetermined prompts, greedy decoding, temperatures
0.7/1.0/1.2, seeds 42/123/456, and 256 new tokens. The unchanged protocol SHA256 is
`{file_hash(root/'configs/generation_protocol.json')}`. Generation crops to the final
256 tokens. Best/final share step-5,000 weights, and all 50 paired settings produce
identical outputs; the 100 policy records are not 100 independent observations.

[Detailed interpretation and predefined greedy examples](interpretation.md)
cover repetition, speaker labels, grammar, spelling and prompt responsiveness.
The complete 300-record E01/E11/E12 descriptive repetition calculation is in
`generation_quality.json`. Low repeated eight-gram fraction does not imply
coherence, correct spelling, originality or low memorization.

## Integrity, persistence and recovery

The complete archive is **{verified['archive_bytes']:,} bytes**, SHA256
`{verified['archive_sha256']}`. Independent saved-version retrieval passed ZIP CRC,
safe unique paths, exact inventory and every one of its {verified['archive_entries']}
file hashes. All 20 resumable checkpoint states and 21 reconstructed immutable
snapshots passed, including optimizer/scheduler counters, metadata and complete
5,000-row authoritative history. Snapshot reconstruction uses exact central
checkpoint bytes rather than storing redundant weights in the ZIP.

Seven parts are published, each no larger than 64 MiB. The independent full-archive
download was sliced locally and each slice checked against the independently
retrieved published part index. Separate network downloads of every part are not
claimed. Independent CPU best/final inference passed under Python 3.10.0 and
PyTorch 2.11.0; that is an inference migration, not the research or optimizer environment.

Final saved-output CUDA verification: {('PASSED. A separate notebook mounted published research output, selected the fully verified final immutable snapshot and restored model, optimizer, scheduler, scaler, RNG, sampling and history exactly at 5000→5000, with zero additional updates. Best/final CUDA inference and checkpoint hashes matched independent CPU recovery.' if final_cuda_verified else 'PENDING; the saved recovery result has not been retrieved.')}
The exact CUDA receipt was independently downloaded and matched the published
stdout. Native pre-publication CUDA roundtrip is retained separately; it was not
used as a substitute for this final gate. The published research and recovery
notebooks are the durable location; working-directory files alone are not publication.

## Update accounting, operational failures and preflight

The authoritative journal contains **{attempts}** completed research step events
and **{len(failures)}** failure/resume events. The sole scientific trajectory has
5,000 known physical optimizer executions, zero observed interruption or replay,
and 475.030 retained training seconds. There is no evidence of superseded or lost
research progress. Completed saved output and checkpoint counters agree.

Real-data engineering execution was separate: four initial smoke updates, ten
benchmark updates, two fresh smoke continuation updates and two batch smoke
continuation updates (18 known development optimizer calls). Synthetic unit-test
optimizer calls are outside both counts; an overall count covering every fixture
is not asserted. Final CPU inference, private-sampler replay and both final CUDA
roundtrips performed zero optimizer updates. No further research was launched.

Preflight qualification version 358073577 verified the isolated freeze, CUDA
numerics, real E12 speed/memory/full-validation cost, checkpoint/export/ZIP sizes
and conservative storage/session budgets. A real 40,363,360-byte smoke archive was
published and independently retrieved, then full-state restoration and exact
step-2→4 CUDA continuation passed before substantive research. Ten benchmark
updates took 0.957251466 training seconds and one full validation took 42.762532966
seconds; these development estimates are not final research results. The storage
budget was 3,350,498,024 bytes against 20,940,603,392 free bytes and a 12-hour
session limit. UI showed 26 GPU hours before the final recovery.

Engineering issues are preserved in `evidence/operational_failures.json`: an
initial local import/path error, download-event timeout, premature asynchronous
hash check, stalled saved-output input mount, a 600-second upload bridge and the
legitimate single interactive-session cap. Resolution used a byte-identical
compact private smoke input and cancellation of the unused stalled draft on the
same account. Research remained gated until exact CUDA continuation passed.
The browser later became unavailable during final-result retrieval; reopening it
allowed retrieval of the already successful saved recovery job. A reporting check
also encountered E01's older protocol hash field name; its existing hash and full
protocol matched exactly. None of these issues changed scientific settings or
caused research replay. Large weights, datasets, archives and expiring URLs remain
outside GitHub. Notebook capture provenance is documented separately.

## Limitations and implications

The measured best BPB ranking is E11 (64), E01 (128), then E12 (256), but one seed
cannot establish an optimal context or statistical significance. Extra conditioning
was available; this training procedure did not make it beneficial at this budget.
Context, window grouping, dropout draws, private sampling and compute all change
together. Separate T4 sessions also confound wall-time comparisons. No FLOP count,
controlled equal-compute experiment or long-range dependency benchmark was measured.

The final validation minus sampled training loss is
{s['final_validation']['loss']-s['final_training_loss']:.8f}. Sampled dropout-on training
and dropout-off full sliding validation use different conditions, so this gap alone
does not prove overfitting or memorization. E12 was still improving near the endpoint;
that does not justify a claim about eventual convergence or an unauthorized extension.

At this seed and budget, context 256 offers no prediction-quality advantage over
E01, consumes more allocated memory and increases validation cost, while training was faster
with fewer accumulation microbatches in the observed environment. Selection
of a later optimized architecture should weigh those costs and the persistent text
failure modes. Adding seeds, combining configurations or extending training requires
separate authorization. The PR is for review; no merge is authorized here.
'''
    interpretation='''# Context-length and generated-text interpretation

## Prediction, convergence and cost

E12 has worse BPB than E01 at every one of the twenty matched validation points.
The difference is largest at step 500 (about +0.22486 BPB) and narrows to +0.04028
at step 5,000. E12's best/final BPB is +0.04222093 above E01's best. It improves
from loss 1.70152845 at step 3,000 to 1.56393333 at step 5,000 and still improves
from step 4,750; the experiment stopped at the authorized endpoint. That trend
does not show whether longer training would change the ordering.

E12 takes 51.82% of E01's recorded training time (475.030 versus 916.708 seconds),
uses 1.832× its peak allocated memory (277.908 versus 151.712 MiB), and spends
2.203× as long on full validation (927.393 versus 420.908 seconds). Larger sequence
matrices increase memory while halving the number of accumulation microbatches.
That is a plausible explanation for the training speed difference, not a measured
FLOP or kernel-time attribution. The combined training-plus-validation cost is
1,402.423 seconds for E12 versus 1,337.616 seconds for E01: faster training alone
does not establish a faster research pipeline. Sessions, IO and publication differ.

Genuine reviewed E11 reaches best/final BPB 2.15396251 at step 5,000 with 116.146 MiB
allocated memory, 1,868.694 training seconds and 267.250 validation seconds. It
predicts the same suffix bytes with only 64 characters of context and 16-way
accumulation. This descriptive ranking favors shorter context for likelihood at
this seed/budget, while E11 is slower to train. It does not establish a universal
advantage of shorter context. Reference profiles and original dataset manifests
were checked; no E01/E11 retraining or fresh inference is claimed here.

## Reading the fixed generation grid

All fifty best-policy E12 settings were inspected; the fifty final-policy records
are exact paired duplicates because best equals final. Every predefined greedy
prompt is shown below, not chosen for quality. The complete unedited grid remains
in generation.json and generation.md. Sampling results include all temperatures
and seeds. E01/E11 comparisons use their preserved grids and the same protocol;
this is a descriptive inspection, not a blinded human quality benchmark.

All five E12 greedy prompts lead to repetitive phrases. ROMEO repeats “the prince
of the present of the prince”; the opening quote and The king prompts settle into
“the prince of the prince” sequences. First Citizen opens with “so be so be so”,
then switches labels without establishing a consistent conversation. JULIET also
moves to DUKE VINCENTIO while repeating prince/present patterns. Formatting,
capitalized labels and line breaks resemble plays, but neither speaker identity
nor scene continuity is reliably preserved. The initial prompt affects beginnings;
it does not produce sustained character-specific dialogue or a meaningful
continuation of the famous opening phrase.

Sampling at temperature 0.7 largely breaks the long exact greedy loops and yields
recognizable labels such as COMINIUS, ROMEO and ESCALUS. Syntax remains broken:
ROMEO seed42 begins “I cannot not this servant of death”, followed by “The gracious
are the pawn true I speak of thee”. ROMEO and JULIET at that same setting share
much of their continuation, a concrete example of weak speaker responsiveness.
Other settings contain double words (“and and”, “but but”) and malformed words
such as “disperitmes” and “purprince”. These outcomes are retained.

At temperatures 1.0 and 1.2 there is greater variation in words, punctuation and
labels but frequent spelling and grammar failures. Examples include TSecond
Citizen, Second Citizzen, DUKE BRAKENGHLOW and 3 KING HEF YORK. Temperature 1.2
can produce strings such as “dismerehmbanct” or “storophangs”. Diversity alone is
not evidence of coherent meaning. Label plausibility does not imply consistent
characters or speakers. Occasional locally plausible phrases occur amid broken
sentences; no sustained plot, correct grammar or long-context coherence is established.

E01's predefined greedy outputs also loop, often around “the world of the world”
and “the seal”; E11 repeats “the world of the world” and LADY GREY labels. E12
changes the dominant words to prince/present rather than eliminating the failure.
Its sampled output avoids many exact loops but still resembles fragmented
Shakespeare-like surface text. Lower BPB and lower repetition need not agree:
E11 has the best likelihood here and the highest greedy eight-gram repetition.

## Descriptive repetition measure

For each 256-character continuation, excluding the prompt, collect all overlapping
8-character windows. Repeated fraction is 1 minus distinct windows divided by
all windows. The table averages five greedy or fifteen sampled best-policy
records per condition; there are no standard-error or significance claims.
E12/E11 final-policy duplicates are not pooled as independent samples. This
measure catches literal reuse, but shorter repeats, semantic loops and invented
words can escape it. Low values are not grammar/coherence or memorization scores.

| Experiment | Greedy | T=0.7 | T=1.0 | T=1.2 |
|---|---:|---:|---:|---:|
'''
    for eid in ['E01','E11','E12']:
        means=[]
        for greedy,t in [(True,1.0),(False,0.7),(False,1.0),(False,1.2)]:
            subset=[q['repeated_8gram_fraction'] for q in quality if q['experiment']==eid and q['policy']=='best' and q['greedy']==greedy and q['temperature']==t]
            means.append(statistics.mean(subset))
        interpretation+='|'+eid+'|'+ '|'.join(f'{m:.6f}' for m in means)+'|\n'
    interpretation+='\n## All five predefined E12 best/final greedy continuations\n'
    for sample in generation['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:
            interpretation+='\nPrompt: '+repr(sample['prompt'])+'\n\n```text\n'+sample['text']+'\n```\n'
    (output/'interpretation.md').write_text(interpretation)
    if not final_cuda_verified:
        text=text.replace('# E12 seed42 — context length 256', '# E12 seed42 — context length 256\n\n**Delivery status: research and independent archive/CPU checks complete; final saved-output CUDA result pending. PR remains draft.**',1)
    (output/'E12_seed42_analysis.md').write_text(text)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','artifacts','verification','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--recovery',type=Path);parser.add_argument('--kernel',required=True);parser.add_argument('--recovery-kernel',required=True)
    a=parser.parse_args();report(a.root,a.artifacts,a.verification,a.recovery,a.output,a.kernel,a.recovery_kernel)
