"""Render E09 measured evidence and validated three-way depth comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    source=a.artifacts;out=a.output;out.mkdir(parents=True,exist_ok=False)
    s=read(source/'analysis/summary.json');c=read(source/'completion.json');d=read(a.diagnostics)
    comparison=read(a.comparison/'comparison.json');gpu=read(a.recovery);local=read(source/'independent_recovery.json')
    manifest=read(a.manifest);timing=read(a.platform);env=read(source/'environment.json');hw=read(source/'hardware.json');review=read(a.review)
    audit=read(source/'depth_audit.json');io=read(source/'io_timing.json');g=read(source/'analysis/generation.json');quality=read(a.quality)
    assert s['completed_optimizer_updates']==5000 and s['parameters']==3584256
    assert c['validation_steps']==list(range(250,5001,250)) and audit['block_count']==8 and len(audit['checks'])==15 and audit['device']=='cuda'
    assert len(read(source/'snapshot_index.json'))==21
    assert comparison['comparability_passed'] and d['journal_steps']==5000 and d['failures']==d['resumes']==0
    assert local['optimizer_updates']==5000 and local['verified_file_count']==len(manifest['files'])
    assert gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    assert g['protocol']==read(a.root/'configs/generation_protocol.json') and len(g['samples'])==100
    assert quality['generation_file_sha256']['E09']==hashlib.sha256((source/'analysis/generation.json').read_bytes()).hexdigest()
    shutil.copyfile(a.quality,out/'generation_quality.json')
    for name in ['summary.json','generation.json','generation.md','validation.csv']:shutil.copyfile(source/'analysis'/name,out/name)
    shutil.copytree(a.comparison,out/'comparison');(out/'figures').mkdir();(out/'evidence').mkdir()
    for p in (source/'analysis/figures').iterdir():
        if p.suffix in {'.png', '.pdf'}:shutil.copyfile(p,out/'figures'/p.name)
    receipts={'artifact_manifest.json':a.manifest,'cuda_recovery.json':a.recovery,'checkpoint_diagnostics.json':a.diagnostics,'platform_timing.json':a.platform,'execution_review.json':a.review}
    for name in ['completion','environment','hardware','execution_command','dataset_validation','depth_audit','storage_preflight','preflight','io_timing','wall_time','snapshot_index','independent_recovery']:receipts[name+'.json']=source/(name+'.json')
    for name,p in receipts.items():shutil.copyfile(p,out/'evidence'/name)
    cp=sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint')
    exports=sum(r['seconds'] for r in io['records'] if r['operation']=='export')
    norms=next(r for r in comparison['gradients'] if r['experiment']=='E09')
    observations={'updates':5000,'validations':20,'training_steps_per_second':5000/s['training_seconds'],'tokens_per_second':s['training_tokens']/s['training_seconds'],'checkpoint_seconds':cp,'export_seconds':exports,'journal_seconds':io['journal_seconds'],'generation_seconds':g['seconds'],'archive_seconds':manifest['archive_seconds'],'pipeline_seconds':timing['pipeline_wall_seconds'],'gradient_statistics':norms}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    text=f'''# E09 seed42: eight-block Transformer research experiment

One explicitly authorized run: 5,000 completed optimizer updates and 20 full sliding
validations. No other seed/configuration/extension was launched. Prior reports
and E07 work are preserved. The broader three-seed study remains separately gated.

## Research question and prior hypothesis

How do eight blocks change next-character likelihood, optimization, cost and
generation relative to four/six blocks? Additional sequential transformations and
capacity might improve or worsen outcomes; improvement was not assumed. E09 adds
1779456 parameters versus E01 (98.59574468%) and889728 versus E08. Depth, capacity,
initialization RNG consumption and compute change together, so this is not an
isolated depth-causality experiment. Observations and interpretation are separated.

## Exact provenance and architecture audit

- Merged preparation main:d1ca09df3d20478aabcdd3b4999f8cef6e0d5782 (E08 PR12).
- Clean trained source:`{s['source_commit']}`.
- Scientific code hash:`{s['code_hash']}`; equivalent audited E01 source/configs.
- Resolved configuration hash:`{s['config_hash']}`; unique run:`{s['run_id']}`.
- [Private research saved notebook v1](https://www.kaggle.com/code/{hw['kernel_id']}).
- [Fresh CUDA recovery v1](https://www.kaggle.com/code/{a.recovery_kernel}).
- GPU:{env['gpu']},device0,total{hw['devices'][0]['total_bytes']:,} bytes;
  {hw['gpu_count']} GPUs allocated, one training device.
- Python{env['python']},torch{env['torch']},torch CUDA{env['cuda']};OS{env['platform']}.
- Exact freeze, driver information and source state are retained in evidence.

Character vocabulary 65, width 192, eight blocks, six heads of dimension 32,
context 128, MLP expansion 4, GELU, sinusoidal positions, dropout 0.1 and
pre-LayerNorm; **3,584,256 trainable parameters**. No RoPE/learned positions.
The15 CPU/CUDA numerical checks include eight independent parameter allocations,
preLN residual reference computations, Q/K/V shape(2,6,128,32), sqrt32 causal
softmax/value/merge192 reference, finite manual next-token CE, future invariance,
finite/nonzero gradients in all eight blocks, train/eval dropout and generation
cropping. Full audits and block-gradient measurements are in evidence.
Smoke gradients establish connectivity only, not research vanishing/exploding
behavior. Shared initialization equality is recorded rather than presumed; the
first four blocks/token embedding match E01, while output-head weights differ
after extra blocks consume RNG. All research weights were initialized freshly.

## Locked training, data and evaluation

AdamW with LR 0.0003, constant scheduler, weight decay 0.01, clipping norm 1,
float32 CUDA, seed 42, microbatch 4, accumulation 8, 4,096 targets per update,
exactly 5,000 updates. No AMP,
early stopping, LR/protocol change or previous trained weights. Independent
development smoke steps precede full reseeding and are not research observations.

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e.
Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed;
character canonical descriptor SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
Source/partition/tokenizer/manifest fields matched audited reference. Contiguous
train/validation/test1003854/55770/55770 bytes; train-only fitting/windows,
no partition-crossing windows or test likelihood/selection.

Full sliding validation every 250 updates: all 55,770 validation bytes, 256
conditioning-only bytes, 55,514 scored original target bytes exactly once,
dropout-free,context128,batch32. BPB=sumNLL/(ln2*scored bytes); character
BPB=loss/ln2 and perplexity=exp(loss). BPB and NLL are related measures.
Positions reset within cropped128-token windows as in the canonical evaluator.
Training losses are sampled/dropout-on; gaps against sliding validation are
unequal-protocol diagnostics. No invented step-zero evaluation or capped
research validation. All20 validation states and21 snapshots are preserved.

## Tests actually executed

| Verification | Actual result |
|---|---|
| Eight-block CPU numerical audit | 15 checks passed |
| Eight-block CUDA numerical audit | 15 checks passed |
| Focused orchestration, depth and generation tests | {review['focused_suite']['passed']} passed in {review['focused_suite']['seconds']:.2f} s |
| Final Windows full suite | {review['final_windows_suite']['passed']} passed, {review['final_windows_suite']['skipped']} CUDA-only skip in {review['final_windows_suite']['seconds']:.2f} s |
| Actual Kaggle CUDA full suite | {review['actual_Kaggle_suite']['passed']} passed in {review['actual_Kaggle_suite']['seconds']:.2f} s |
| Real-data development CUDA preflight | Training, validation, save/load, export/restore, ZIP CRC, headless plotting and generation passed |
| Published archive recovery | Exact file set, all file hashes, ZIP CRC, best/final CPU inference passed |
| Fresh compatible CUDA recovery | All saved training states restored, zero extra optimizer updates |

The development smoke update and test optimizer updates are separate from the
5,000-update research history. Exact executed commands and evidence receipts
are linked from interpretation.md.

## Actual measured results

| Quantity | Value |
|---|---:|
| First training loss,step1 | {s['initial_recorded_training']['loss']:.8f} |
| First validation loss,step250 | {s['initial_recorded_validation']['loss']:.8f} |
| Final training loss | {s['final_training_loss']:.8f} |
| Final100-step trailing mean | {s['final_training_trailing_100_loss']:.8f} |
| Best step | {s['best_validation']['step']} |
| Best loss/BPB | {s['best_validation']['loss']:.8f}/{s['best_validation']['bpb']:.8f} |
| Final loss/BPB | {s['final_validation']['loss']:.8f}/{s['final_validation']['bpb']:.8f} |
| Final perplexity | {s['final_validation']['perplexity']:.8f} |
| Processed targets/source bytes | {s['training_tokens']:,}/{s['training_source_bytes']:,} |
| Unique target-byte coverage | {s['unique_target_bytes']:,} |
| Peak allocated GPU MiB | {s['peak_gpu_bytes']/2**20:.3f} |

Repeated byte exposure is not independent examples/epochs. Peak allocated
memory excludes reserved/device/driver overhead. Missing target coverage is
{1003854-s['unique_target_bytes']} bytes; the first source byte lacks a prior target context.

| Update | E01 loss | E08 loss | E09 loss | E01 BPB | E08 BPB | E09 BPB |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in comparison['matched_validation']:text+=f"| {r['step']} | "+' | '.join(f"{r[k]:.8f}" for k in ['E01_loss','E08_loss','E09_loss','E01_bpb','E08_bpb','E09_bpb'])+' |\n'
    text+='\n## Three-way measured depth/capacity comparison\n\n| Metric | E01 | E08 | E09 |\n|---|---:|---:|---:|\n'
    for metric,key,sub in [('Parameters','parameters',None),('Best loss','best_validation','loss'),('Best BPB','best_validation','bpb'),('Best step','best_validation','step'),('Final BPB','final_validation','bpb'),('Training seconds','training_seconds',None),('Validation seconds','validation_seconds',None),('Peak allocated bytes','peak_gpu_bytes',None)]:
        text+=f'| {metric} | '+' | '.join(f"{v[key][sub] if sub else v[key]:.8f}" for v in comparison['summaries'].values())+' |\n'
    text+='\n| Reference | Candidate | Metric | Absolute delta | Relative,% |\n|---|---|---|---:|---:|\n'
    for r in comparison['pairwise_differences']:text+=f"| {r['reference']} | {r['candidate']} | {r['metric']} | {r['delta']:.8f} | {r['relative_percent']:.3f} |\n"
    text+='\nCollector/schema/config/environment/data/tokenizer compatibility passed for all\nthree genuine5000-step histories. Private sampling RNG and coverage arrays match.\nEqual update/token counts do not equal FLOPs or wall time. Sample SD is undefined\nfor one seed, not zero. No statistical significance or test-selected winner.\n\n'
    for p in sorted((out/'comparison').glob('*.png')):text+=f'![Measured three-way comparison: {p.stem}](comparison/{p.name})\n\n'
    rope=read(a.root/'report/pilots/E03_seed42/summary.json')
    text+=f"Separate contextual E03 RoPE best BPB{rope['best_validation']['bpb']:.8f},parameters{rope['parameters']}: a position ablation, not a depth control or combined-architecture result.\n\n"
    text+='## Actual optimization descriptors\n\n| Experiment | Median norm | p95 | Maximum | Clipped updates |\n|---|---:|---:|---:|---:|\n'
    for r in comparison['gradients']:text+=f"| {r['experiment']} | {r['median']:.8f} | {r['p95']:.8f} | {r['maximum']:.8f} | {r['clipped_updates']}/5000 |\n"
    text+=f"\nAggregate pre-clipping norms only; no longitudinal layerwise research gradients. Actual3k-to5k loss improvement{s['loss_improvement_3000_to_5000']:.8f}; descriptive observed-point slope{s['late_validation_slope_per_1000_steps']:.8f} per1000 updates. See interpretation.md for convergence and generalization conclusions.\n\n"
    text+='## Runtime scopes\n\n| Measured scope | Seconds |\n|---|---:|\n'
    for label,value in [('Training only',s['training_seconds']),('20 full validations',s['validation_seconds']),('Checkpoint calls',cp),('Snapshot export calls',exports),('Journal flush/fsync',io['journal_seconds']),('Training CLI wall',io['cli_wall_seconds']),('100-output generation',g['seconds']),('Archive/hash scope',manifest['archive_seconds']),('Clone-to-archive pipeline',timing['pipeline_wall_seconds']),('Local verification/CPU inference',local['seconds'])]:text+=f'| {label} | {value:.3f} |\n'
    text+=f"\nTraining-only{observations['training_steps_per_second']:.4f} updates/s,{observations['tokens_per_second']:.2f} targets/s. Pipeline excludes queue, final conversion/publication, download and fresh GPU recovery wall time. Inner measured scopes overlap the pipeline; do not sum twice. Individual validation duration and dollar costs were not recorded. Separate T4 sessions confound speed comparisons.\n\n## Genuine E09 figures\n\n"
    for p in sorted((out/'figures').glob('*.png')):text+=f'![Measured E09: {p.stem}](figures/{p.name})\n\n'
    text+='## Complete fixed generation protocol\n\n[All100 unedited samples](generation.md) and generation.json preserve best/final,\nfive prompts,greedy,temperatures.7/1/1.2,seeds42/123/456,256 new tokens.\nFive predetermined best greedy outputs follow; no external text editing/model.\n\n'
    if c['best_checkpoint']==c['final_checkpoint']:text+='Best/final are the same weight state; labeled grids are duplicate policies, not independent draws.\n\n'
    for sample in g['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:text+=f"### {sample['prompt']!r}\n\n`{sample['checkpoint']}`,greedy,seed42,256 new tokens.\n\n```text\n{sample['text']}\n```\n\n"
    text+=f'''## Durable publication and independent recovery

Archive{manifest['archive_bytes']:,} bytes; SHA256`{manifest['archive_sha256']}`.
All{local['verified_file_count']} file hashes, exact file set, safe paths and ZIP CRC
passed independently. Best/final CPU inference and fresh compatible CUDA inference
passed. Full model/optimizer/scheduler/scaler/RNG/sampling/history state restored
at 5,000 with zero additional updates. All 20 checkpoint counters independently
verified; 21 immutable snapshots stay in published outputs. Exact checkpoint/file
hashes are in evidence/artifact_manifest.json and snapshot_index.json. Large files
are outside Git on D: and in published Kaggle outputs. No research failure/resume event.

**Best:** `{c['best_checkpoint']}`. **Final:** `{c['final_checkpoint']}`.

## Interpretation and limitations

[Detailed observations, qualitative analysis, recommendations and executed/retrieval commands](interpretation.md)
complete this report. Hypothesis is separate from measurement; one-seed results
cannot establish significance or depth-only causality. Context128, one dataset,
initialization/dropout RNG differences, extra capacity and separate GPU-session
conditions constrain interpretation. Character BPB and NLL are mathematically
related; generation repetition/phrase overlap do not establish memorization or
understanding. No unapproved extensions or test selection occurred.
'''
    (out/'E09_seed42_analysis.md').write_text(text,encoding='utf-8')
    write_interpretation(out,s,comparison,d,quality,review,manifest,source,a.recovery_kernel)
    (out/'evidence/receipt_checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'evidence').glob('*.json') if p.name!='receipt_checksums.json'},indent=2)+'\n')
    print(out/'E09_seed42_analysis.md')


def write_interpretation(out,s,comparison,diagnostics,quality,review,manifest,source,recovery_kernel):
    summaries=comparison['summaries'];baseline=summaries['E01'];six=summaries['E08']
    percent=lambda candidate,reference:100*(candidate/reference-1)
    groups=quality['groups']
    q=lambda policy,temp,greedy:next(r['mean_repeated_8gram_position_fraction'] for r in groups if r['experiment']=='E09' and r['checkpoint_policy']==policy and r['temperature']==temp and r['greedy']==greedy)
    intervals=diagnostics['improvement_intervals'];overlaps=diagnostics['generation_diagnostics']
    e09grad=next(r for r in comparison['gradients'] if r['experiment']=='E09')
    better={eid:sum(r['E09_loss']<r[eid+'_loss'] for r in comparison['matched_validation']) for eid in ['E01','E08']}
    text=f'''# Interpretation, scientific limits and reproducible recovery

This accompanies [the measured E09 report](E09_seed42_analysis.md). The report,
tables and figures describe one authorized E09 seed-42 run, not a multi-seed
result. No other research configuration, seed or extension was launched.

## Hypothesis, observations and interpretation

The prior hypothesis was that additional Transformer blocks might change
predictive accuracy and optimization, while increasing capacity and cost.
Improvement was not assumed. Only n_layers changes from four to eight; the
model adds 1,779,456 parameters (98.5957%). Compared with six-block E08 it adds
889,728 parameters. Width, tokenizer, context, optimizer, token budget,
validation and precision are identical. This isolates the configuration override,
but does not isolate depth independently of capacity or computation.

**Observation:** E09 has lower validation loss at {better['E01']}/20 matching
checkpoints versus E01 and {better['E08']}/20 versus E08. Best loss is
{s['best_validation']['loss']:.8f} at update {s['best_validation']['step']}; best
conditional BPB is {s['best_validation']['bpb']:.8f}. Best BPB changes by
{percent(s['best_validation']['bpb'],baseline['best_validation']['bpb']):.3f}%
against E01 and {percent(s['best_validation']['bpb'],six['best_validation']['bpb']):.3f}%
against E08. All twenty exact matched observations and absolute/relative deltas
are in comparison/matched_validation.csv and comparison/pairwise_differences.csv.

**Interpretation:** Additional depth/capacity benefits held-out next-character
likelihood for this seed and update budget. Character BPB is loss divided by
ln(2), so lower NLL and BPB are related evidence, not independent achievements.
No statistical significance, depth-only causality, general superiority or
improved understanding is established. One run per model cannot estimate
across-seed SD; fluctuations across optimizer updates are not seed uncertainty.

## Convergence and generalization

The first training loss is recorded at update 1, the first validation at 250.
There is no invented step-zero validation. Observed interval improvements are:

| Interval | Validation NLL decrease | Decrease per 1,000 updates |
|---|---:|---:|
'''
    for r in intervals:text+=f"| {r['start']}–{r['end']} | {r['loss_decrease']:.8f} | {r['decrease_per_1000_steps']:.8f} |\n"
    text+=f'''
**Observation:** The net 3k–5k loss improvement is
{s['loss_improvement_3000_to_5000']:.8f}. Improvement is diminishing over successive
1,000-update intervals, but has not ceased. The best scheduled point is 4,750;
the final loss rises by {s['final_validation']['loss']-s['best_validation']['loss']:.8f}
({percent(s['final_validation']['loss'],s['best_validation']['loss']):.3f}%). The final
loss remains below the 4,500 measurement. Minima between scheduled evaluations
are unknown. The late observed-point fitted slope is descriptive, not a forecast.

**Interpretation:** One final uptick does not establish sustained overfitting or
double descent. The trailing 100-update training mean is
{s['final_training_trailing_100_loss']:.8f}, giving a diagnostic final validation
gap of {s['final_validation']['loss']-s['final_training_trailing_100_loss']:.8f} nats.
Training uses sampled windows with dropout; validation uses complete sliding
windows without dropout. The gap therefore is not a direct memorization rate.
It is larger than E08's corresponding gap, but one-seed uncertainty remains.
No test likelihood or test-based architecture selection was performed.

## Measured cost and compute efficiency

E09 uses {s['training_seconds']:.3f} training-only seconds,
{s['validation_seconds']:.3f} full-validation seconds and
{s['peak_gpu_bytes']/2**20:.3f} MiB peak allocated GPU memory. Training time changes
by {percent(s['training_seconds'],baseline['training_seconds']):.3f}% versus E01 and
{percent(s['training_seconds'],six['training_seconds']):.3f}% versus E08. Parameter
count and actual training time are distinct from the fixed 20,480,000 target-token
budget. All three final private sampling RNG states and target-byte coverage
arrays match. Repeated target exposure is not independent examples or epochs.
Only {s['unique_target_bytes']:,} of 1,003,854 training bytes were covered as targets.

At E01-like training cost, E09's observed 2,500-step point used 907.084 seconds
and had loss 1.58724586, while E01 ended at loss 1.53601210 after 916.708 seconds.
These are actual nearby observations, not an interpolated crossing or a controlled
equal-runtime experiment. Extra depth is more effective per update here, but
does not establish better predictive performance per GPU second. The nine
comparison plots include BPB versus updates, cumulative measured training time,
parameter count and allocated memory. Separate T4 sessions and execution noise
limit speed comparisons. Allocated memory excludes reserved/driver/device costs.
No dollar cost or FLOP measurement was recorded.

The aggregate pre-clipping gradient median is {e09grad['median']:.8f}, p95
{e09grad['p95']:.8f}, maximum {e09grad['maximum']:.8f}; clipping threshold 1 was
exceeded on {e09grad['clipped_updates']}/5,000 updates
({100*e09grad['clipped_updates']/5000:.2f}%). Norms and losses are finite throughout.
These are aggregate gradients; per-layer longitudinal research gradients were
not recorded. CPU/CUDA smoke connectivity across all eight blocks does not prove
the absence of vanishing/exploding gradients during all research updates.

## Complete genuine generation evaluation

[All 100 unedited outputs](generation.md) preserve both checkpoint policies,
five fixed prompts, greedy decoding, temperatures 0.7/1.0/1.2, seeds 42/123/456
and 256 new characters. The measured report shows all five predetermined best
greedy samples, including their failures. No external model edited these outputs.

The fraction of repeated 8-gram positions averages {q('best',1.0,True):.6f} for
E09 best greedy outputs, versus E01 0.522892 and E08 0.620080. Final E09 greedy
averages {q('final',1.0,True):.6f}, versus E01 0.543775 and E08 0.562249.
These descriptive fixed-grid values suggest less exact repetition in E09's best
greedy grid than E08's, but do not establish coherent language or a general
decoding improvement. Best temperature means are
{q('best',0.7,False):.6f}/{q('best',1.0,False):.6f}/{q('best',1.2,False):.6f}
for 0.7/1.0/1.2. Low exact repetition does not imply correct grammar.

**Observed failures:** Best greedy prompts repeatedly converge to “Second
Murderer”, “What shall he will have the present of the prince?” and repeated
“so”. JULIET produces “the seat of the seat” and “the season of the season”.
Dialogue labels, line breaks and familiar local words are recognizable, while
semantic continuity, grammar and prompt-specific character behavior remain weak.
Temperature sampling reduces long exact loops but produces malformed words such
as “modtal”, “promper”, “meddital”, “latements” and “pospering”. Samples with
speaker labels and less repetition still often have inconsistent syntax and
little longer-distance meaning. These are illustrative observations from the
fixed grid, not an automatic coherence rating or statistically significant gain.

Exact training-text overlap was checked for ten predetermined best samples
(all five greedy and all five temperature-0.7/seed-42 outputs). Their longest
matching substrings range from {min(r['longest_exact_training_substring_characters'] for r in overlaps)}
to {max(r['longest_exact_training_substring_characters'] for r in overlaps)} characters.
See evidence/checkpoint_diagnostics.json for exact strings and offsets. Ordinary
Shakespearean phrases repeat; this subset diagnostic is not a comprehensive
memorization audit or proof of copied training passages. No test text was used.

## Preservation, recovery and failures

The research saved notebook version 1 completed. The {manifest['archive_bytes']:,}-byte
archive passed SHA256, exact file set, safe extraction, ZIP CRC and all 59 file
hashes independently. Best/final CPU inference passed. Fresh compatible CUDA
recovery `{recovery_kernel}` restored model, optimizer, scheduler, scaler, RNG,
sampling and history exactly at 5,000, with zero additional updates. Its recovered
best/final checkpoint hashes match the archive manifest. All twenty optimizer
and scheduler counters were independently checked. Twenty-one immutable snapshots
are retained in published `pilot/snapshots`; the ZIP omits their duplicate files
but includes every resumable run checkpoint plus the snapshot index/manifests.

Large files are on D: and in saved Kaggle outputs, excluded from Git. The local
verified run is `{source}`. Checkpoint hashes and retrieval commands are recorded
with the manifest and executed-command evidence. Both published versions are
private to the legitimate rayyannmerchant account.

The local read-only log stream required reconnects and temporarily returned 401
after OAuth access expiration. Official existing-account OAuth refresh restored
access; the user also confirmed private reconnection. These were monitoring
failures, not trainer restarts. No research failure or resume event appears in
the 5,000-step journal. No credential, installed SDK patch, quota workaround or
uncommitted Windows training code was used. Headless plotting and mounted
read-only recovery inputs worked; the earlier E01 plotting issue did not recur.

## Recommendation and remaining limits

Keep 5k as an initial screening budget, with separately approved 10k extensions
of selected configurations considered informative. E09 improves through 4,750
and has a nontrivial 3k–5k gain, so 5k does not prove convergence. The final uptick
and generation loops make a longer controlled comparison worth investigating,
not automatically better. First prioritize additional independently authorized
seeds and matched provenance; no extension is launched by this recommendation.

E03 RoPE's contextual best BPB 2.07327076 at 1,804,800 parameters remains better
than this E09 observation. It is a separate positional ablation, not a depth
control or evidence for an untested combined architecture. Single dataset,
single seed, extra capacity, initialization/dropout RNG consumption, finite
context and separate GPU-session costs constrain every conclusion. The first
four block/token initialization weights match E01, while the output head changes
because extra block initialization consumes RNG. Identical numeric seeds do not
mean identical whole-model initialization. Across-seed SD remains undefined.

The authorized E09 run is complete; the broader three-seed study remains
separately gated/NO-GO. Main and all earlier experiments are preserved. E08 PR12
was owner-merged before preparation; no formal review submission was returned
and none is claimed. This task opens/updates E09 PR13 for review and does not
merge it. E07's independent work is untouched.

## Reproduction evidence

- Cloud training: evidence/execution_command.json, using unchanged shared CLI.
- Source and actual CPU/CUDA tests: evidence/execution_review.json and audits.
- Local verification/analysis commands: evidence/commands.json.
- Final report integrity checks: evidence/final_checks.json.
- Exact archive file hashes: evidence/artifact_manifest.json.
- CPU recovery: evidence/independent_recovery.json.
- Fresh zero-update CUDA recovery: evidence/cuda_recovery.json.
- Measured timing scopes: evidence/platform_timing.json and evidence/io_timing.json.
- Retrieval instructions: [artifact_retrieval.md](artifact_retrieval.md).

All figures and tables derive from actual recorded histories. No unobserved
validation point, synthetic research measurement, fabricated SD or successful
unexecuted test is presented.
'''
    (out/'interpretation.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifacts','comparison','platform','recovery','manifest','diagnostics','quality','review','root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-kernel',required=True);report(p.parse_args())
