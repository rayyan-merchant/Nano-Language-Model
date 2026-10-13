"""Render verified context-ablation evidence; no training or invented measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    source=a.artifacts
    s=read(source/'analysis/summary.json'); completion=read(source/'completion.json')
    comparison=read(a.comparison/'comparison.json'); diagnostics=read(a.diagnostics)
    recovery=read(a.recovery); manifest=read(a.manifest)
    local=read(source/'independent_recovery.json'); env=read(source/'environment.json')
    hardware=read(source/'hardware.json'); audit=read(source/'context_audit.json')
    preflight=read(source/'preflight.json'); generation=read(source/'analysis/generation.json')
    quality=read(a.quality); timing=read(a.platform); io=read(source/'io_timing.json')
    assert s['parameters']==1804800 and s['completed_optimizer_updates']==5000
    assert completion['validation_steps']==list(range(250,5001,250))
    assert len(read(source/'snapshot_index.json'))==21
    assert audit['status']=='passed' and audit['device']=='cuda' and len(audit['checks'])==14
    assert preflight['context_length']==64 and preflight['accumulation']==16
    assert preflight['sampled_windows_per_update']==64 and preflight['predicted_positions_per_update']==4096
    assert comparison['comparability_passed'] and comparison['only_model_factor_difference']=='context_length'
    assert len(comparison['sampling_audits'])==2
    assert diagnostics['journal_steps']==5000 and diagnostics['failures']==diagnostics['resumes']==0
    assert recovery['optimizer_updates_before']==recovery['optimizer_updates_after']==5000
    assert recovery['additional_optimizer_updates']==0 and all(recovery['exact_state'].values())
    assert local['archive_sha256']==manifest['archive_sha256'] and local['verified_file_count']==len(manifest['files'])
    assert len(generation['samples'])==100 and generation['protocol']==read(a.root/'configs/generation_protocol.json')
    assert quality['generation_file_sha256']['E11']==hashlib.sha256((source/'analysis/generation.json').read_bytes()).hexdigest()
    out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'evidence').mkdir()
    shutil.copytree(a.comparison,out/'comparison');shutil.copytree(source/'analysis/figures',out/'figures')
    for name in ['summary.json','validation.csv','generation.json','generation.md']:
        shutil.copyfile(source/'analysis'/name,out/name)
    shutil.copyfile(a.quality,out/'generation_quality.json')
    receipts={'artifact_manifest.json':a.manifest,'checkpoint_diagnostics.json':a.diagnostics,
              'cuda_recovery.json':a.recovery,'platform_timing.json':a.platform}
    for name in ['completion','environment','hardware','dataset_validation','context_audit','preflight',
                 'storage_preflight','full_validation_preflight','execution_command','io_timing','wall_time',
                 'snapshot_index','independent_recovery']:
        receipts[name+'.json']=source/(name+'.json')
    for name,path in receipts.items():shutil.copyfile(path,out/'evidence'/name)
    observations={'training_updates_per_second':5000/s['training_seconds'],
                  'targets_per_training_second':s['training_tokens']/s['training_seconds'],
                  'checkpoint_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'),
                  'export_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='export'),
                  'journal_seconds':io['journal_seconds'],'generation_seconds':generation['seconds'],
                  'archive_seconds':manifest['archive_seconds'],'pipeline_seconds':timing['pipeline_wall_seconds']}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf-8')
    text=f'''# E11 seed42: 64-token causal context experiment

## Scope, objective and hypothesis

One explicitly authorized E11 seed42 run: 5,000 completed updates, twenty full
sliding validations. No additional seeds, E12, hybrid or extension were launched.
The broader three-seed study remains separately gated. Earlier evidence is preserved.
Question: how does shortening causal context128→64 affect prediction,
optimization, runtime, memory and generated text? Less available preceding text
may remove useful dependencies; shorter tensors may reduce computation, but
sixteen accumulation microbatches may increase overhead. Neither outcome was assumed.

## Architecture and correctness

Only context_length changes. Parameters remain **1,804,800**: vocabulary65,
width192, four independent preLN blocks, six heads/head32, MLP192→768→192,
GELU, sinusoidal positions and dropout0.1. No learned positions or RoPE.
The14 development CPU/CUDA checks verify sole override, seed42 weight equality,
sinusoidal0–63 matching E01's prefix, shifted64-token targets, finite CE,
64×64 causal masks, independent explicit sqrt32 Q/K/V reference, future
invariance, overlength rejection, all finite nonzero parameter gradients,
train/eval dropout, repeated generation cropping64, sequential sliding evaluator
reference and analytic character BPB. Numerical fixtures are development evidence;
they are not Tiny Shakespeare measurements. Full real-data CUDA preflight is
recorded separately in evidence/full_validation_preflight.json.

Seed42 produces identical initial trainable tensors for E01 and E11 because the
shortened sinusoidal buffer consumes no initialization RNG. Training trajectories
still differ through window shapes, dropout draws and private sampling progression.

## Reproducible source and environment

- Synchronized reviewed main:317ba670112ffa3123c4bc26baddf9fa1059c60c, including E10PR14/E07PR11.
- Clean trained source:`{s['source_commit']}`.
- Scientific hash:`{s['code_hash']}`; canonical package unchanged.
- Run:`{s['run_id']}`; configuration hash:`{s['config_hash']}`.
- [Kaggle research version1](https://www.kaggle.com/code/{hardware['kernel_id']}).
- [Fresh zero-update recovery version1](https://www.kaggle.com/code/{a.recovery_kernel}).
- GPU:{env['gpu']}, training device0, {hardware['devices'][0]['total_bytes']:,} available device bytes;
  {hardware['gpu_count']} devices attached, one used for training.
- Python{env['python']}, torch{env['torch']}, CUDA{env['cuda']}, NumPy1.26.4.
- OS:{env['platform']}. Exact qualified transitive freeze and driver output preserved.

Native Kaggle packages were untouched. No CPU fallback, AMP or uncommitted
Windows training source was used. Scientific algorithms were not modified;
the thin wrapper derives context/accumulation and checks resolved settings.

## Data, training and validation method

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e;
source SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed;
character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
All manifest/source/tokenizer/partition fields matched. Vocabulary65; original
train/validation/test bytes1,003,854/55,770/55,770. Train-only tokenizer fitting,
no crossing training windows, no test likelihood or test-based selection.

Fresh seed42; AdamW, constant LR0.0003, weight decay0.01, clipping1.0,
float32 CUDA, microbatch4, accumulation16, exactly4096 targets/update and5000
updates. No early stopping or automatic extension. The separate one-update
smoke does not count toward research updates; canonical training reseeds afterward.

Full sliding validation every250: exclude common first256 bytes from scoring,
score all55,514 suffix bytes exactly once, context64, batch32, dropout disabled.
The256-byte exclusion is not attention context: each prediction sees at most
its preceding64 tokens, with local sinusoidal positions reset in each crop.
BPB=sumNLL/(ln2×scored bytes), character BPB=loss/ln2, perplexity=exp(loss).
Loss/BPB are comparable to E01 on identical targets under different intended
conditioning restrictions. No capped research validation or step-zero fiction.

## Token budget, sampling and optimization confound

| Run | Context | Microbatch | Accumulation | Windows/update | Targets/update |
|---|---:|---:|---:|---:|---:|
| E01 |128|4|8|32|4096|
| E11 |64|4|16|64|4096|

E11 samples320,000 windows versus E01's160,000. Both predict20,480,000 target
positions/source bytes, but window grouping, maximum start index and private
sampling RNG progression differ. Independent replay regenerated each saved
private RNG and exact source-target union; equality between runs was not required.
Equal token exposure does not imply equal FLOPs, contextual information or
gradient grouping. This is an optimization/computation confound beyond the
intended context restriction. Target-byte coverage is not input traffic or epochs.

```json
{json.dumps(comparison['sampling_audits'],indent=2)}
```

## Genuine measured results

| Quantity | Value |
|---|---:|
| First recorded training loss/update1 |{s['initial_recorded_training']['loss']:.8f}|
| First recorded validation loss/update250 |{s['initial_recorded_validation']['loss']:.8f}|
| Final training loss |{s['final_training_loss']:.8f}|
| Final100-update mean |{s['final_training_trailing_100_loss']:.8f}|
| Best checkpoint update |{s['best_validation']['step']}|
| Best loss/BPB |{s['best_validation']['loss']:.8f}/{s['best_validation']['bpb']:.8f}|
| Final loss/BPB |{s['final_validation']['loss']:.8f}/{s['final_validation']['bpb']:.8f}|
| Final perplexity |{s['final_validation']['perplexity']:.8f}|
| Training seconds |{s['training_seconds']:.3f}|
| Full-validation seconds |{s['validation_seconds']:.3f}|
| Pipeline seconds |{observations['pipeline_seconds']:.3f}|
| Updates/training second |{observations['training_updates_per_second']:.5f}|
| Targets/training second |{observations['targets_per_training_second']:.3f}|
| GPU peak allocated MiB |{s['peak_gpu_bytes']/2**20:.3f}|
| Repeated predicted targets/source bytes |{s['training_tokens']:,}/{s['training_source_bytes']:,}|
| Unique target-byte coverage |{s['unique_target_bytes']:,}|

Peak allocation excludes reserved/driver memory. Pipeline duration is the
measured notebook execution interval through archive construction; queue,
publication, network transfer and independent recovery are separate. Training
excludes validation/IO. Separately measured overheads:

```json
{json.dumps(observations,indent=2)}
```

## All twenty matched validations

| Update | E01 loss | E11 loss | Δloss | E01 BPB | E11 BPB |
|---|---:|---:|---:|---:|---:|
'''
    for r in comparison['matched_validation']:
        text+=f"|{r['step']}|{r['E01_loss']:.8f}|{r['E11_loss']:.8f}|{r['delta_loss']:.8f}|{r['E01_bpb']:.8f}|{r['E11_bpb']:.8f}|\n"
    text+='\n## E01 comparison and resource costs\n\n| Metric | E01 | E11 | Δ E11−E01 | Relative % |\n|---|---:|---:|---:|---:|\n'
    for r in comparison['summary_comparison']:
        text+=f"|{r['metric']}|{r['E01']:.8f}|{r['E11']:.8f}|{r['delta_E11_minus_E01']:.8f}|{r['relative_percent']:.3f}|\n"
    text+='\nLower loss/BPB is better. Speed and memory are observed in separate T4 sessions; no controlled repeated timing or statistical significance is claimed. Same parameter count does not imply equal computation.\n'
    for p in sorted((out/'comparison').glob('*.png')):
        text+=f'\n![Observed comparison: {p.stem}](comparison/{p.name})\n'
    for p in sorted((out/'figures').glob('*.png')):
        text+=f'\n![Measured E11: {p.stem}](figures/{p.name})\n'
    text+=f"\n## Convergence and gradient observations\n\nObserved3k–5k loss decrease:{s['loss_improvement_3000_to_5000']:.8f}; recorded-point slope:{s['late_validation_slope_per_1000_steps']:.8f} per1000 updates; final-minus-best:{s['final_validation']['loss']-s['best_validation']['loss']:.8f}. These describe scheduled measurements, not interpolated observations or guaranteed future convergence.\n\n```json\n{json.dumps(comparison['gradient_statistics'],indent=2)}\n```\n\nAggregate norms are pre-clipping; no longitudinal per-layer gradient measurements or vanishing/exploding-gradient claim. Training uses dropout-on sampled windows; validation uses dropout-off full sliding targets. Their gap is not a direct memorization rate. Single-seed fluctuations cannot prove overfitting.\n"
    text+='\n## Complete genuine text generation\n\n[All100 unedited outputs](generation.md) include best/final, five fixed prompts, greedy and temperatures0.7/1/1.2 with seeds42/123/456,256 new tokens. Context64 enforced at every step. No output editing or favorable-only selection. Five predetermined best greedy examples follow.\n'
    for sample in generation['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:
            text+=f"\n### Prompt {sample['prompt']!r}\n\nCheckpoint:{sample['checkpoint']}; seed{sample['seed']}; greedy;256 new tokens.\n\n```text\n{sample['text']}\n```\n"
    text+='\nGeneration quality statistics and100 matched E01/E11 settings are preserved in generation_quality.json. Repeated8-grams describe loops, not grammatical quality. Exact train-text overlap in checkpoint_diagnostics.json is a descriptive memorization indicator, not proof. Context-related failures must be interpreted alongside sampling/optimization differences. See interpretation.md for observations from the actual full grid.\n'
    text+=f"\n## Durable publication and independent recovery\n\nArchive:{manifest['archive_bytes']:,} bytes, SHA256`{manifest['archive_sha256']}`. Independently retrieved archive:CRC/exact inventory/{len(manifest['files'])} file hashes verified. All20 resumable validation checkpoint states and21 immutable snapshots verified. Best/final recovered CPU inference passed. Fresh compatible T4 restoration preserved every model/optimizer/scheduler/scaler/RNG/sampling/training-state field with5000→5000 updates; zero extra research updates. Constant LR retains its regime for a later separately approved extension.\n\nLarge artifacts are outside Git. Local retrieval root:`{source}`. Durable source:the private research version1 linked above; snapshots remain in its saved outputs. Complete run/logs/metadata/generation/figures are in E11_seed42_pilot.zip with artifact_manifest.json. Use scripts/pilot/download_kaggle_archive.py and inspect_archive.py; fresh CUDA recovery uses build_notebook.py --restore-only --source-kernel with exact trained source. Never treat /kaggle/working alone as permanent.\n\n## Failures, evidence boundaries and recommendations\n\nJournal reports{diagnostics['failures']} research failures and{diagnostics['resumes']} resumes. Transport/platform/setup evidence and actual commands are recorded in evidence; later operational issues must remain explicit. No invented missing metrics or standard deviation.\n\nInterpretation and budget recommendation must follow actual late validation trend and matched comparisons, not the assumption that shorter context is faster or weaker. One dataset and one seed cannot establish generalization or significance. The future context framework contains genuine E11/E01 results and E12=null/NOT_AUTHORIZED_NOT_RUN. No E12 launch or10k extension. Additional seeds/context256 require separate approval.\n"
    (out/'E11_seed42_analysis.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','artifacts','comparison','diagnostics','recovery','manifest','quality','platform','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-kernel',required=True)
    report(p.parse_args())
