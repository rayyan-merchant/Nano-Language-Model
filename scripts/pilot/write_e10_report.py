"""Render only verified E10 measurements; no training or synthetic outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    source=a.artifacts;s=read(source/'analysis/summary.json');c=read(source/'completion.json')
    compared=read(a.comparison/'comparison.json');d=read(a.diagnostics);gpu=read(a.recovery)
    local=read(source/'independent_recovery.json');m=read(a.manifest);env=read(source/'environment.json')
    hw=read(source/'hardware.json');io=read(source/'io_timing.json');g=read(source/'analysis/generation.json')
    audit=read(source/'mlp_audit.json');review=read(a.review);timing=read(a.platform);quality=read(a.quality)
    windows=review.get('final_windows_suite',review['windows_suite'])
    assert s['parameters']==1213440 and s['completed_optimizer_updates']==5000 and s['run_id'].startswith('E10_seed42_')
    assert c['validation_steps']==list(range(250,5001,250)) and len(read(source/'snapshot_index.json'))==21
    assert audit['device']=='cuda' and audit['status']=='passed' and len(audit['checks'])==14
    assert compared['comparability_passed'] and compared['only_model_factor_difference']=='mlp_expansion'
    assert d['journal_steps']==5000 and d['failures']==d['resumes']==0
    assert gpu['optimizer_updates_before']==gpu['optimizer_updates_after']==5000 and gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    assert local['verified_file_count']==len(m['files']) and local['archive_sha256']==m['archive_sha256']
    assert g['protocol']==read(a.root/'configs/generation_protocol.json') and len(g['samples'])==100
    assert quality['generation_file_sha256']['E10']==hashlib.sha256((source/'analysis/generation.json').read_bytes()).hexdigest()
    out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'evidence').mkdir()
    shutil.copytree(a.comparison,out/'comparison');shutil.copytree(source/'analysis/figures',out/'figures')
    for name in ['summary.json','validation.csv','generation.json','generation.md']:shutil.copyfile(source/'analysis'/name,out/name)
    shutil.copyfile(a.quality,out/'generation_quality.json')
    receipts={'artifact_manifest.json':a.manifest,'checkpoint_diagnostics.json':a.diagnostics,'cuda_recovery.json':a.recovery,'execution_review.json':a.review,'platform_timing.json':a.platform}
    for name in ['completion','environment','hardware','dataset_validation','mlp_audit','preflight','storage_preflight','execution_command','io_timing','wall_time','snapshot_index','independent_recovery']:receipts[name+'.json']=source/(name+'.json')
    for name,path in receipts.items():shutil.copyfile(path,out/'evidence'/name)
    cp=sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint');export=sum(r['seconds'] for r in io['records'] if r['operation']=='export')
    observations={'training_updates_per_second':5000/s['training_seconds'],'predicted_tokens_per_training_second':s['training_tokens']/s['training_seconds'],'checkpoint_seconds':cp,'export_seconds':export,'journal_seconds':io['journal_seconds'],'generation_seconds':g['seconds'],'archive_seconds':m['archive_seconds'],'pipeline_seconds':timing['pipeline_wall_seconds'],'efficiency_tradeoffs':compared['efficiency_tradeoffs']}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf-8')
    text=f'''# E10 seed42: 2× MLP expansion research experiment

One authorized run: exactly 5,000 completed optimizer updates and 20 full sliding
validations. No other configuration, seed, hybrid or extension was launched.
Earlier reports and the collaborator's E07 PR11 are preserved. The broader
three-seed study remains separately gated/NO-GO.

## Research objective and prior hypothesis

How does reducing feed-forward expansion from 4× to 2× affect prediction,
optimization, efficiency and text quality? The smaller MLP could retain much of
the baseline's predictive quality with lower cost, or lose useful capacity.
Neither improvement nor regression was presumed. Architecture and capacity
change together; this does not isolate parameter count alone.

## Architecture and numerical audit

Only `mlp_expansion` changes. Four independent pre-LayerNorm blocks each contain
Linear(192,384), GELU and Linear(384,192), totaling 148,032 MLP parameters per
block instead of 295,872. Total **1,213,440 parameters**, 591,360 fewer than E01
(32.765957%). Width192, six heads/head32, context128, sinusoidal positions,
dropout0.1 and explicit causal attention remain unchanged. No RoPE or learned
positions. No prior experiment weights were loaded.

The 14 CPU/CUDA checks cover the sole override/training defaults, deterministic
seed42 construction, independent block storage, both projection dimensions and
counts, a separate F.linear/GELU MLP reference, preLN residual references,
explicit Q/K/V reshape/sqrt32/causal softmax/value aggregation, sinusoidal-only
positions, shifted next-token CE, future invariance, finite gradients reaching
both MLP projections in every block, dropout train/eval and cropped generation.
Actual audit receipts include projection gradient norms and compatible-shape
initialization equality. Changed tensor sizes consume initialization/dropout RNG
differently; equal seeds do not imply equal whole-model weights. Initialization
algorithms are unchanged. Smoke gradients establish connectivity, not longitudinal
research gradient behavior.

## Exact source and qualified environment

- Synchronized main: a6b28d7d97f9739dddd24ec4c7c3765bc65265dc, after E09 PR13 merge.
- Clean trained source: `{s['source_commit']}`.
- Audited scientific hash: `{s['code_hash']}`.
- Resolved configuration hash: `{s['config_hash']}`; run: `{s['run_id']}`.
- [Private Kaggle research saved version1](https://www.kaggle.com/code/{hw['kernel_id']}).
- [Fresh zero-update CUDA recovery version1](https://www.kaggle.com/code/{a.recovery_kernel}).
- GPU {env['gpu']}, device0; {hw['gpu_count']} devices allocated, one used for training;
  total GPU0 {hw['devices'][0]['total_bytes']:,} bytes.
- Python {env['python']}, PyTorch {env['torch']}, CUDA {env['cuda']}, NumPy1.26.4.
- OS {env['platform']}; exact freeze, driver and hardware metadata in evidence.

Qualified transitive freeze matches E01 except editable source SHA. Native
Kaggle packages were untouched. No uncommitted Windows training source or CPU
fallback was used. Separate T4 sessions confound precise speed comparisons.

## Data and fixed training/evaluation protocol

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e.
Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77,
vocabulary65. All source/partition/tokenizer/manifest reference fields matched.
Contiguous train/validation/test bytes1,003,854/55,770/55,770; train-only fitting,
no crossing windows, no test likelihood or test-based checkpoint selection.

AdamW LR0.0003 constant, weight decay0.01, clipping1, float32 CUDA, seed42,
microbatch4, accumulation8, 4,096 targets/update, exactly5,000 updates. No AMP,
early stopping or protocol change. Independent development preflight precedes
full reseeding. Full sliding validation every250 scores exactly55,514 original
bytes after256 conditioning-only bytes, dropout disabled, context128/batch32.
Conditional BPB=sumNLL/(ln2×scored bytes); character BPB=loss/ln2 and
perplexity=exp(loss). These are related measures, not independent achievements.
Local positions reset in cropped windows as in E01. No capped research validation
or invented step-zero measurement. Training uses sampled/dropout-on windows;
validation gaps are unequal-protocol diagnostics, not memorization rates.

## Tests actually executed

- CPU/CUDA MLP audits: 14 checks each passed.
- Focused suite: {review['focused_suite']['passed']} passed in {review['focused_suite']['seconds']:.2f}s.
- Final Windows suite: {windows['passed']} passed, {windows['skipped']} CUDA-only skip in {windows['seconds']:.2f}s.
- Actual Kaggle CUDA suite: {review['actual_Kaggle_suite']['passed']} passed in {review['actual_Kaggle_suite']['seconds']:.2f}s.
- Real-data CUDA smoke: training/validation/save/load/export/restore/ZIP/Agg/generation passed.
- All20 optimizer/scheduler checkpoint counters, journals and BPB identity passed.
- Archive CRC/all-file hashes/CPU inference and fresh full-state CUDA restore passed.

The first Windows full-suite attempt had20 fixture setup errors because its new
D-drive parent did not exist. The parent was created and the full rerun passed;
this was resolved before submission, without changing scientific code. Actual
commands and final integrity checks are in evidence/commands.json and final_checks.json.

## Measured results

| Quantity | Actual value |
|---|---:|
| First training loss, update1 | {s['initial_recorded_training']['loss']:.8f} |
| First validation loss, update250 | {s['initial_recorded_validation']['loss']:.8f} |
| Final training loss | {s['final_training_loss']:.8f} |
| Final100-update training mean | {s['final_training_trailing_100_loss']:.8f} |
| Best validation update | {s['best_validation']['step']} |
| Best validation loss/BPB | {s['best_validation']['loss']:.8f}/{s['best_validation']['bpb']:.8f} |
| Final validation loss/BPB | {s['final_validation']['loss']:.8f}/{s['final_validation']['bpb']:.8f} |
| Final perplexity | {s['final_validation']['perplexity']:.8f} |
| Predicted tokens/source-byte exposure | {s['training_tokens']:,}/{s['training_source_bytes']:,} |
| Unique target-byte coverage | {s['unique_target_bytes']:,} |
| GPU peak allocated MiB | {s['peak_gpu_bytes']/2**20:.3f} |

Repeated target exposure is not unique examples, input traffic, corpus epochs or
FLOPs. Peak allocation excludes reserved/driver memory. No missing value is invented.

## All twenty matched E01/E10 validations

| Update | E01 loss | E10 loss | Δloss | E01 BPB | E10 BPB |
|---|---:|---:|---:|---:|---:|
'''
    for r in compared['matched_validation']:text+=f"| {r['step']} | {r['E01_loss']:.8f} | {r['E10_loss']:.8f} | {r['delta_loss']:.8f} | {r['E01_bpb']:.8f} | {r['E10_bpb']:.8f} |\n"
    text+='\n## Predictive quality and resource tradeoffs\n\n| Metric | E01 | E10 | Absolute delta | Relative,% |\n|---|---:|---:|---:|---:|\n'
    for r in compared['summary_comparison']:text+=f"| {r['metric']} | {r['E01']:.8f} | {r['E10']:.8f} | {r['delta_E10_minus_E01']:.8f} | {r['relative_percent']:.3f} |\n"
    text+='\n```json\n'+json.dumps(compared['efficiency_tradeoffs'],indent=2)+'\n```\n\nBoth final private sampling RNG and byte-coverage arrays match. The collector\nvalidated configuration, code/profile, dataset/tokenizer hashes and complete\nhistories. Lower BPB is better; dividing BPB by parameter count is not an intrinsic\nefficiency score. Interpret paired quality/cost and explicit marginal tradeoffs,\nwith single-seed and separate-session caveats. Capacity and MLP architecture change\ntogether. SD is undefined for one seed, not zero; no significance or superiority\nclaim follows merely from a smaller/faster model.\n\n'
    for p in sorted((out/'comparison').glob('*.png')):text+=f'![Measured E01/E10 comparison: {p.stem}](comparison/{p.name})\n\n'
    text+='## Actual optimizer and gradient descriptors\n\n| Experiment | Median norm | p95 | Maximum | Clipped updates |\n|---|---:|---:|---:|---:|\n'
    for r in compared['gradient_statistics']:text+=f"| {r['experiment']} | {r['median']:.8f} | {r['p95']:.8f} | {r['maximum']:.8f} | {r['clipped_updates']}/5000 |\n"
    text+='\nAggregate pre-clipping norms only. These do not establish layerwise vanishing/exploding gradients; finite smoke gradients establish connectivity. AdamW, constant LR0.0003 and clipping threshold1 remained fixed.\n\n## Runtime scopes\n\n| Measured scope | Seconds |\n|---|---:|\n'
    for label,value in [('Training only',s['training_seconds']),('20 full validations',s['validation_seconds']),('Checkpoint calls',cp),('Snapshot export calls',export),('Journal flush/fsync',io['journal_seconds']),('Training CLI wall',io['cli_wall_seconds']),('100-output generation',g['seconds']),('Archive/hash scope',m['archive_seconds']),('Clone-to-archive pipeline',timing['pipeline_wall_seconds']),('Independent local CPU verification/inference',local['seconds'])]:text+=f'| {label} | {value:.3f} |\n'
    text+=f"\nTraining-only {observations['training_updates_per_second']:.4f} updates/s and {observations['predicted_tokens_per_training_second']:.2f} targets/s. Inner scopes overlap the pipeline and must not be added twice. Pipeline excludes queue, final notebook conversion/publication, network download and fresh recovery. Their combined end-to-end wall time, individual validation durations, FLOPs and dollar cost were not recorded.\n\n## Genuine E10 learning/optimization figures\n\n"
    for p in sorted((out/'figures').glob('*.png')):text+=f'![Measured E10: {p.stem}](figures/{p.name})\n\n'
    text+=f"## Convergence descriptors\n\nObserved3k–5k NLL improvement {s['loss_improvement_3000_to_5000']:.8f}; late observed-point slope {s['late_validation_slope_per_1000_steps']:.8f} per1,000 updates. Best scheduled update {s['best_validation']['step']}; final minus best {s['final_validation']['loss']-s['best_validation']['loss']:.8f}. This is descriptive, not a forecast or interpolation of missing measurements. See interpretation.md for actual trends and limitations.\n\n## Complete genuine generation protocol\n\n[All100 unedited outputs](generation.md) preserve five fixed prompts, best/final, greedy, temperatures0.7/1/1.2, seeds42/123/456 and256 new tokens. Five predetermined best greedy samples follow, including poor outputs. No external model generated or improved text.\n\n"
    for sample in g['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:text+=f"### {sample['prompt']!r}\n\nCheckpoint `{sample['checkpoint']}`,greedy,seed42,256 new tokens.\n\n```text\n{sample['text']}\n```\n\n"
    if c['best_checkpoint']==c['final_checkpoint']:text+='Best and final are the same weight state. Their separately labeled grids duplicate the same fixed settings, not independent draws or two independently trained models.\n\n'
    text+=f"## Durable preservation and independent recovery\n\nArchive {m['archive_bytes']:,} bytes; SHA256 `{m['archive_sha256']}`. Exact entry set, safe paths, ZIP CRC and all {local['verified_file_count']} file hashes passed independently. Best/final CPU and fresh CUDA inference passed. All20 checkpoint counters and21 immutable snapshots are verified. Full optimizer/scheduler/scaler/RNG/sampler/history restored at5,000 with zero extra updates. Archive retains all20 resumable states; duplicate immutable snapshot files remain in published outputs. Large artifacts stay on D: and Kaggle, excluded from Git.\n\nBest: `{c['best_checkpoint']}`. Final: `{c['final_checkpoint']}`. Exact per-file hashes are in [artifact manifest](evidence/artifact_manifest.json). No research failure/resume event appears in the journal.\n\n## Interpretation, limitations and next-stage implications\n\n[Detailed observations, genuine generation analysis and recommendations](interpretation.md) complete this report. [Retrieval instructions](artifact_retrieval.md) locate large evidence. One seed/dataset, finite context, changed capacity/initialization RNG and separate GPU sessions constrain interpretation. Phrase overlap/repetition do not prove memorization or understanding. Additional seeds and10k extensions require separate approval; none was launched.\n"
    (out/'E10_seed42_analysis.md').write_text(text,encoding='utf-8')
    write_interpretation(out,s,compared,d,quality,read(a.root/'report/pilots/E01_seed42/summary.json'))
    print(out/'E10_seed42_analysis.md')


def write_interpretation(out,s,comparison,d,quality,baseline):
    pct=lambda x,y:100*(x/y-1)
    groups=quality['groups'];repeat=lambda eid,temp,greedy:next(r['mean_repeated_8gram_position_fraction'] for r in groups if r['experiment']==eid and r['checkpoint_policy']=='best' and r['temperature']==temp and r['greedy']==greedy)
    nbetter=sum(r['E10_loss']<r['E01_loss'] for r in comparison['matched_validation'])
    grad=next(r for r in comparison['gradient_statistics'] if r['experiment']=='E10')
    text=f'''# Interpretation, efficiency limits and reproducible evidence

This supplements [the measured E10 report](E10_seed42_analysis.md). All results
describe one explicitly authorized seed-42 run of 5,000 updates. No additional
configuration, seed, hybrid model or extension was executed. Hypotheses,
observations, interpretations and recommendations are distinguished below.

## Hypothesis and controlled architecture change

A smaller MLP might retain useful predictive quality with lower memory and
compute requirements, or lose useful capacity. No direction of effect was
assumed. E10 changes only mlp_expansion from four to two, hidden width768→384,
and removes591,360 parameters (32.765957%). Attention width192, six heads,
four blocks, context128, character tokenizer, GELU, sinusoidal positions and
dropout0.1 remain fixed. The explicit Q/K/V attention and pre-LayerNorm/residual
implementation is unchanged. This is an MLP architecture-plus-capacity ablation,
not an isolation of parameter count or a universal scaling-law experiment.

The 14 actual CPU/CUDA numerical checks verify all four MLP shapes and counts,
manual linear/GELU/residual/attention references, finite shifted CE and gradients,
future invariance, dropout and generation cropping. Changing tensor dimensions
changes default initialization scales where fan-in changes, RNG consumption and
subsequent initial weights; the initialization algorithm itself was not modified.
Smoke equality receipts explicitly record compatible-shape comparisons rather
than presuming whole-model equality under the same numeric seed. The private
data-sampling generator and final coverage arrays match E01 exactly.

## Predictive observations and interpretation

**Observation:** Best and final loss are {s['final_validation']['loss']:.8f}, BPB
{s['final_validation']['bpb']:.8f}, at5,000. E10 beats E01 at {nbetter}/20 matched
validation boundaries. Best BPB is {pct(s['best_validation']['bpb'],baseline['best_validation']['bpb']):.3f}%
above E01's best; final BPB is {pct(s['final_validation']['bpb'],baseline['final_validation']['bpb']):.3f}%
above E01's final. Exact absolute/relative deltas and all twenty measurements are
in comparison/matched_validation.csv and summary_comparison.csv. Character
loss/BPB are directly comparable over identical original scored bytes, but are
mathematically related, not independent evidence. Token perplexities are not
compared with BPE experiments. Test likelihood remains reserved.

**Interpretation:** Under this seed, data and update budget, reduced feed-forward
capacity/architecture has a predictive penalty. The magnitude should be weighed
against resource constraints; no preregistered tolerance defines “nearly the
same”, so the report does not declare equivalence. Nor is the model better simply
because it is smaller. One seed cannot establish significance, depth/width
generalization, parameter-count-only causality or absence of seed sensitivity.
Across-seed SD is undefined, not zero. Earlier depth/position experiments are
different single-factor comparisons, not evidence for an untested hybrid model.

## Convergence and generalization

| Observed interval | Validation NLL decrease |
|---|---:|
'''
    for r in d['improvement_intervals']:text+=f"| {r['start']}–{r['end']} | {r['loss_decrease']:.8f} |\n"
    text+=f'''
**Observation:** Validation improves to the final scheduled checkpoint. The
3k–5k decrease is {s['loss_improvement_3000_to_5000']:.8f}; the4k–5k decrease is
{next(r['loss_decrease'] for r in d['improvement_intervals'] if r['start']==4000):.8f}.
The late observed-point fitted slope is {s['late_validation_slope_per_1000_steps']:.8f}
nats per1,000 updates. It is descriptive, not a forecast of a10k result. There
is no invented step-zero validation or inferred intermediate minimum.

**Interpretation:** Improvements diminish but have not ceased; the endpoint
does not demonstrate complete convergence. No sustained validation deterioration
is observed, so overfitting/double descent is not established. A smaller model
could have a capacity limit or simply need further optimization; this run does
not distinguish those hypotheses. Final100-update training mean is
{s['final_training_trailing_100_loss']:.8f}, versus validation
{s['final_validation']['loss']:.8f}, gap
{s['final_validation']['loss']-s['final_training_trailing_100_loss']:.8f} nats.
Sampled dropout-on training and full dropout-off sliding validation have different
measurement conditions; this is a diagnostic gap, not a memorization rate.

## Efficiency: measured quality/resource tradeoffs

**Observation:** Training-only cost is {s['training_seconds']:.3f}s,
{pct(s['training_seconds'],baseline['training_seconds']):+.3f}% versus E01. Twenty
full validations cost {s['validation_seconds']:.3f}s,
{pct(s['validation_seconds'],baseline['validation_seconds']):+.3f}% versus E01.
Peak allocated memory is {s['peak_gpu_bytes']/2**20:.3f}MiB,
{pct(s['peak_gpu_bytes'],baseline['peak_gpu_bytes']):+.3f}% versus E01. The smaller
parameter count does not deliver an observed training-speed improvement here.
Full validation is faster and allocated memory is lower. These are separately
measured scopes, not interchangeable throughput claims.

**Interpretation:** Predictive quality per update is worse, while model storage
and run memory requirements are lower. Quality versus training-only cost also
does not improve in this observation. The relative training difference is small
and was measured in separate T4 sessions, so it is not a robust timing advantage
or regression. We did not profile kernel utilization or CPU overhead; explanations
for the flat training time would be hypotheses, not measured causes. Evaluation
uses batch32 versus training microbatch4/accumulation8, so their measured effects
need not match. No isolated per-token deployment latency, FLOP count or monetary
cost was recorded. Generation-grid wall time includes loading and sampling and
is not an isolated inference benchmark.

The comparison stores an explicit marginal BPB penalty per million parameters
removed, paired quality/cost observations and targets per training second.
Dividing BPB by parameter count is not an intrinsic model-quality score. A compact
model can be useful under storage/memory constraints without being the best
predictor or fastest trainer. No resource weighting or equivalence tolerance was
predeclared, so no single arbitrary ratio is used to pronounce an overall winner.

## Optimization descriptors

All recorded losses and aggregate gradient norms are finite. E10 norm median is
{grad['median']:.8f}, p95 {grad['p95']:.8f}, maximum {grad['maximum']:.8f}; clipping
threshold1 was exceeded on {grad['clipped_updates']}/5,000 updates
({100*grad['clipping_fraction']:.2f}%). AdamW, LR0.0003 constant, weight decay0.01,
float32 and effective4,096 target tokens/update remained fixed. All20 optimizer
and scheduler counters were checked independently. Per-layer longitudinal
research gradient traces were not recorded; finite nonzero smoke gradients in
each MLP do not prove absence of vanishing/exploding gradients throughout training.

## Genuine generation observations and failures

[All100 unedited outputs](generation.md) retain five fixed prompts, both checkpoint
policies, greedy decoding, temperatures0.7/1/1.2, seeds42/123/456 and256 new tokens.
Best and final are the identical5,000-step state: their separately labeled50-row
grids duplicate the same settings, not independent model draws. The main report
shows all five predetermined best greedy examples, including poor outputs.

**Observation:** Mean repeated8-gram-position fraction for best greedy E10 is
{repeat('E10',1.0,True):.6f}, versus E01 {repeat('E01',1.0,True):.6f}.
All five greedy prompts converge toward “see”, “seem”, “sent” and “sense” loops.
ROMEO and JULIET responses show little persistent character-specific relevance.
Repetition obscures grammatical meaning despite line breaks and recognizable
local words. Temperature0.7 produces Shakespearean-looking speaker labels,
including KING RICHARD III and MENENIUS, without reliable syntax or narrative.
Higher temperatures avoid long exact loops but produce malformed words such as
“lapinish”, “reddistand”, “unorhice” and “kingdrat”. These examples come from the
fixed outputs; text was not rewritten or selected for favorable appearance.

Best temperature repetition fractions for0.7/1/1.2 are
{repeat('E10',0.7,False):.6f}/{repeat('E10',1.0,False):.6f}/{repeat('E10',1.2,False):.6f}.
Low exact repetition is not a grammar/coherence score. No qualitative superiority
or meaningful understanding is established. Identical sampling seeds can yield
shared trajectories under different prompts; the grid is not a set of independent
human-quality ratings.

Exact training-text overlap was checked on ten predetermined best outputs (all
five greedy and all five temperature0.7/seed42 samples). Longest matching
substrings range from {min(r['longest_exact_training_substring_characters'] for r in d['generation_diagnostics'])}
to {max(r['longest_exact_training_substring_characters'] for r in d['generation_diagnostics'])} characters.
Exact strings and training offsets are in evidence/checkpoint_diagnostics.json.
Ordinary phrases repeat; this subset is not a comprehensive memorization audit
or proof that generated passages were copied. No test text was used.

## Preservation, failures and evidence boundaries

Research and fresh CUDA recovery saved version1 completed. Archive retrieval
passed full SHA256, exact entry set, ZIP CRC, safe extraction and every file hash.
Best/final CPU inference and fresh CUDA full-state roundtrip passed; model,
optimizer, scheduler, scaler, RNG, sampling and histories restored at5,000 with
zero additional updates. All20 checkpoints and21 immutable snapshots remain
published. The ZIP stores checkpoints once; duplicate snapshot files stay in
saved outputs. Large evidence remains on D: and Kaggle, outside Git.

The native research journal has no failure/resume event. Read-only SSE monitoring
required reconnects, without restarting training. Initial Windows test setup
failed due to a missing D-drive parent, then passed after directory creation.
The local generation comparison initially rejected E10 due to a reporting
allowlist omission; the guard and parameterized regression were fixed without
changing trained scientific code or outputs. A later default-temp permission
error was resolved by explicit D-drive basetemp. Two standard CLI archive
downloads ended in IncompleteRead; failed attempts were preserved. The unmodified
official Kaggle API supplied the authorized output URL. A streaming attempt
retained45MiB, then four bounded HTTP-range streams recovered the remaining
bytes. Exact206/Content-Range/size checks and full-archive SHA256 passed before
local CRC/all-file checks. An optional CPU-only fallback submission was rejected
by notebook-title validation before execution and was not retried after local
recovery succeeded; it used no compute. Exact receipts and commands
document these events, rather than presenting them as research restarts.

## Recommendation and optimized-model implications

Retain5k as the initial screening budget. Separately approved additional seeds
and selected10k extensions could distinguish convergence and seed effects, but
none was launched. E10's minimum at the endpoint and continuing3k–5k improvement
make extension informative; they do not prove it would match E01 or justify
silently changing the common budget.

Keep the4× MLP baseline when predictive quality under this training budget is
the priority. Retain2× as a compact candidate where the measured parameter/memory
reduction and validation cost matter enough to accept the observed likelihood
penalty. Additional seeds and explicitly defined resource/quality criteria are
needed before selecting an optimized architecture. Do not combine this change
with RoPE/depth/width or claim an untested hybrid gain. Single seed, one dataset,
finite context, initialization/dropout RNG differences, capacity and separate
GPU sessions constrain the conclusion. The broader study remains gated/NO-GO.

## Reproduction and delivery

[artifact_retrieval.md](artifact_retrieval.md) gives private saved-version URLs,
archive/checkpoint paths and retrieval procedures. evidence/commands.json records
actual execution/analysis commands; evidence/execution_command.json preserves
the unchanged training CLI. Final integrity receipts, tests and file hashes are
in evidence/final_checks.json and receipt_checksums.json. E10 PR14 is reviewable
and not auto-merged. E01–E09 records and the collaborator's E07 PR11 are preserved.
'''
    (out/'interpretation.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifacts','comparison','platform','recovery','manifest','diagnostics','quality','review','root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-kernel',required=True);report(p.parse_args())
