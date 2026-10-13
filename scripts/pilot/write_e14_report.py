"""Render accepted E14 measurements and width comparisons, never train a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prose_spacing(text):
    """Space human prose while preserving every fenced generated/code block."""
    import re
    substitutions={'completed5000':'completed 5,000','Exactly5000':'Exactly 5,000',
                   '5000updates':'5,000 updates','4096targets':'4,096 targets',
                   '32sampled':'32 sampled','20full':'20 full','20 validations':'20 validations',
                   'every250':'every 250','periodic500':'periodic 500','250step':'250 step',
                   '256-character':'256-character','256 new':'256 new','256 common':'256 common',
                   'zeroextra':'zero extra','zero additional':'zero additional','21 immutable':'21 immutable',
                   'all20':'all 20','All20':'All 20','all100':'all 100','All100':'All 100',
                   'all21':'all 21','context128':'context 128','batch32':'batch 32',
                   'float32CUDA':'float32 CUDA','microbatch4':'microbatch 4','accumulation8':'accumulation 8',
                   'MLP240':'MLP 240','width240':'width 240','width144':'width 144',
                   '960parameters':'960 parameters','additionalupdates':'additional updates',
                   'four-character':'four-character','five fixed':'five fixed','preLayerNorm':'pre-LayerNorm'}
    result=[];fenced=False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('```'):
            fenced=not fenced;result.append(line);continue
        if not fenced:
            parts=re.split(r'(`[^`]*`)',line)
            for i in range(0,len(parts),2):
                for old,new in substitutions.items():parts[i]=parts[i].replace(old,new)
                parts[i]=re.sub(r'\bseed42\b','seed 42',parts[i])
                parts[i]=re.sub(r'(?i)(width|context|head|dropout|vocabulary|clipping|decay|training|validation|loss|BPB|temperatures|temperatures|microbatch|accumulation|seed|sinusoidal)(?=\d)',r'\1 ',parts[i])
                parts[i]=re.sub(r'(\d)(parameters|dimensions|tokens|targets|bytes|updates|prompts|seeds|new characters|nats)(?=\b)',r'\1 \2',parts[i])
                parts[i]=parts[i].replace('for5000','for 5,000').replace('full validations','full validations').replace('all LayerNorm dimensions240','all LayerNorm dimensions 240')
            line=''.join(parts)
        result.append(line)
    return ''.join(result)


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    source=a.artifacts;s=read(source/'analysis/summary.json');c=read(source/'completion.json')
    gpu=read(a.recovery);cpu=read(source/'independent_recovery.json');manifest=read(a.manifest)
    pair=read(a.comparison/'comparison.json');widths=read(a.widths/'width_comparison.json')
    diagnostics=read(a.diagnostics);quality=read(a.quality);timing=read(a.platform)
    grid=read(source/'analysis/generation.json');env=read(source/'environment.json');hw=read(source/'hardware.json')
    io=read(source/'io_timing.json');audit=read(source/'width_audit.json');preflight=read(source/'preflight.json')
    development=read(a.development);gate=read(source/'independent_development_recovery_gate.json')
    assert s['parameters']==2808960 and s['completed_optimizer_updates']==5000
    assert c['validation_steps']==list(range(250,5001,250)) and len(read(source/'snapshot_index.json'))==21
    assert audit['status']=='passed' and audit['device']=='cuda' and len(audit['checks'])==20
    assert preflight['model']['d_model']==240 and preflight['accumulation']==8
    assert preflight['sampled_windows_per_update']==32 and preflight['predicted_positions_per_update']==4096
    assert pair['comparability_passed'] and pair['only_model_factor_difference']=='d_model'
    assert len(pair['sampling_audits'])==2 and pair['same_final_sampling_RNG_and_byte_coverage']
    assert diagnostics['journal_steps']==5000 and diagnostics['failures']==diagnostics['resumes']==0
    assert gpu['status']=='fresh_saved_output_CUDA_state_restore_passed_no_updates'
    assert set(gpu['exact_state'])=={'model','optimizer','scheduler','scaler','rng','training_state'}
    assert all(gpu['exact_state'].values()) and gpu['additional_optimizer_updates']==0
    assert gpu['optimizer_updates_before']==gpu['optimizer_updates_after']==5000
    assert gpu['trained_source_commit']==s['source_commit'] and gpu['run_id']==s['run_id']
    assert gpu['environment']['code_hash']==s['code_hash'] and not gpu['environment']['git_dirty']
    assert cpu['archive_sha256']==manifest['archive_sha256'] and cpu['verified_file_count']==len(manifest['files'])
    assert gate['status']=='published_E14_development_fresh_CUDA_full_state_passed' and gate==development
    assert len(grid['samples'])==100 and grid['protocol']==read(a.root/'configs/generation_protocol.json')
    assert quality['generation_file_sha256']['E14']==hashlib.sha256((source/'analysis/generation.json').read_bytes()).hexdigest()
    assert [v['experiment'] for v in widths['resources']]==['E13','E01','E14']
    for x,y in zip(cpu['samples'],gpu['samples']):assert x['sha256']==y['sha256'] and x['text']==y['text']
    out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'evidence').mkdir()
    shutil.copytree(source/'analysis/figures',out/'figures');shutil.copytree(a.comparison,out/'comparison');shutil.copytree(a.widths,out/'width_comparison')
    for name in ['summary.json','validation.csv','generation.json','generation.md']:shutil.copyfile(source/'analysis'/name,out/name)
    shutil.copyfile(a.quality,out/'generation_quality.json')
    receipts={'artifact_manifest.json':a.manifest,'cuda_recovery.json':a.recovery,'checkpoint_diagnostics.json':a.diagnostics,
              'platform_timing.json':a.platform,'published_development_recovery.json':a.development}
    for name in ['completion','environment','hardware','dataset_validation','width_audit','preflight','CUDA_smoke_state_restore',
                 'storage_preflight','full_validation_preflight','execution_command','io_timing','wall_time','snapshot_index',
                 'independent_recovery','independent_development_recovery_gate']:receipts[name+'.json']=source/(name+'.json')
    for name,path in receipts.items():shutil.copyfile(path,out/'evidence'/name)
    observed={'training_updates_per_second':5000/s['training_seconds'],'tokens_per_training_second':s['training_tokens']/s['training_seconds'],
              'checkpoint_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'),
              'export_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='export'),
              'journal_seconds':io['journal_seconds'],'generation_seconds':grid['seconds'],'archive_seconds':manifest['archive_seconds'],
              'clone_to_archive_pipeline_seconds':timing['pipeline_wall_seconds']}
    (out/'observations.json').write_text(json.dumps(observed,indent=2)+'\n',encoding='utf-8')
    base=next(v for v in widths['resources'] if v['experiment']=='E01');candidate=widths['resources'][-1]
    duplicate=c['best_checkpoint']==c['final_checkpoint']
    text=f'''# E14 seed42: width-240 controlled research experiment

## Objective, hypothesis and authorized scope

Question: how does model width192→240 change next-character prediction,
optimization, resource cost and generation on Tiny Shakespeare? Before the run,
additional capacity could improve fitting or cost more without a predictive gain;
no outcome was assumed. One E14 seed42 run was authorized for5000 updates.
Earlier experiments and the collaborator's E12 work were preserved. No additional
research configuration, seed, hybrid or extension was launched.

## Architecture and correctness

The sole intended override is d_model240. Four independent preLayerNorm blocks,
six heads,40dimensions/head, context128, MLP240→960→240, GELU, sinusoidal
positions and dropout0.1. Vocabulary65; embedding65×240; LMhead240→65;
attention/output240→240; all LayerNorm dimensions240. No RoPE, learned positions,
ReLU or BPE. Parameters **2,808,960**, **1,004,160 more** than E01
(55.638298% increase); each MLP has462,000parameters.

Twenty CPU and real CUDA checks verify sole override/defaults, repeat seed42
initialization, independent blocks, projections/counts, manual sqrt40 attention,
causal mask/softmax/value aggregation, preLN residual and linear/GELU references,
shifted next-token CE, finite nonzero gradients across all blocks, future-token
invariance, dropout train/eval behavior, sinusoidal-only positions, overlength
rejection and autoregressive context128 cropping. Development gradient checks
show connectivity, not longitudinal per-layer research gradients.

Width changes capacity, head dimension, MLP width, initialization fan-in/RNG
consumption and positional frequency spacing together. Same seed does not mean
identical initial tensors across widths. No width-only mechanism is established.

## Reproducibility and dataset

- Clean trained commit:`{s['source_commit']}`.
- Scientific hash:`{s['code_hash']}`.
- Run:`{s['run_id']}`; configuration hash:`{s['config_hash']}`.
- [Research saved version1]({a.notebook}).
- [Fresh compatible zero-update recovery]({a.recovery_url}).
- Qualified Tesla T4GPU0; Python{env['python']}, torch{env['torch']}, CUDA{env['cuda']}.
- Qualified NumPy1.26.4 and complete transitive freeze; native Kaggle packages unchanged.
- OS:{env['platform']}; GPU usable memory{hw['devices'][0]['total_bytes']:,}bytes.

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e.
Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
Train/validation/test bytes1,003,854/55,770/55,770; train-only fitting and contiguous
separate windows. All source/split/tokenizer/manifest reference fields matched.
No test likelihood or test selection was performed.

## Training, evaluation and pre-run persistence

Fresh seed42; AdamW0.0003constant, weight decay0.01, clipping1.0, float32CUDA,
microbatch4/accumulation8/4096targets/update/32sampled windows/update. Exactly5000
authoritative updates,20 validations every250, all validation checkpoint states
preserved alongside periodic500 cadence and best-loss selection. No AMP, early
stopping, precision/schedule modification or CPU fallback.

Every validation uses the canonical full sliding evaluator with context128,
batch32/dropoutoff,256 common conditioning-only bytes and55,514 scored original
suffix bytes, each target once. Local positions reset for cropped contexts.
BPB=sum token NLL/(ln2×covered bytes); for this character tokenizer BPB=loss/ln2,
perplexity=exp(loss). No invented step-zero or capped research measurement.

Before research, a separate E14 one-update smoke and10-update development
benchmark were published in a saved output version, independently retrieved and
CRC/SHA256/snapshot-verified. A fresh qualified CUDA runtime restored all six
training-state groups and performed inference at1→1, zero additional updates.
The research wrapper verified that published receipt from its same source.
Development updates are separate from research results. Measured benchmark,
full-validation, checkpoint size, export and storage planning receipts accompany
this report. A short projection is not a measured5000-step runtime.

## Recorded outcomes and time accounting

| Measurement | Value |
|---|---:|
| First training loss/update1 |{s['initial_recorded_training']['loss']:.8f}|
| First validation loss/update250 |{s['initial_recorded_validation']['loss']:.8f}|
| Final training loss |{s['final_training_loss']:.8f}|
| Final100-update training mean |{s['final_training_trailing_100_loss']:.8f}|
| Best update |{s['best_validation']['step']}|
| Best validation loss/BPB |{s['best_validation']['loss']:.8f}/{s['best_validation']['bpb']:.8f}|
| Final validation loss/BPB |{s['final_validation']['loss']:.8f}/{s['final_validation']['bpb']:.8f}|
| Final perplexity |{s['final_validation']['perplexity']:.8f}|
| Training-only seconds |{s['training_seconds']:.3f}|
| Full-validation seconds |{s['validation_seconds']:.3f}|
| Peak allocated GPU MiB |{s['peak_gpu_bytes']/2**20:.3f}|
| Predicted tokens/source-byte exposure |{s['training_tokens']:,}/{s['training_source_bytes']:,}|
| Unique target bytes |{s['unique_target_bytes']:,}|

Training excludes evaluation and IO. Pipeline timing below spans source clone to
archive-complete log events, excluding queue, saved-output publication, network
retrieval and fresh recovery. Allocated memory excludes reserved/driver memory.
Canonical peak telemetry resets once and samples after optimizer updates, before
validation; full-run readings include earlier validation allocations but do not
measure a separate final-validation/postprocessing peak. The short benchmark
reads its peak before its sole final validation, so its memory number cannot
replace the recorded research-run peak. E01/E13/E14 share this measurement policy.
Target bytes count repeated target exposure, not input traffic, unique examples,
epochs or FLOPs. Sampler RNG and target union matched E01 and were independently
replayed; different width/dropout/global RNG consumption need not match.

```json
{json.dumps(observed,indent=2)}
```

## All twenty measured width comparisons

| Update | E13 loss | E01 loss | E14 loss | E13 BPB | E01 BPB | E14 BPB |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in widths['matched_validation']:text+=f"|{r['step']}|{r['E13_loss']:.8f}|{r['E01_loss']:.8f}|{r['E14_loss']:.8f}|{r['E13_bpb']:.8f}|{r['E01_bpb']:.8f}|{r['E14_bpb']:.8f}|\n"
    text+='\n## Capacity, performance and measured resources\n\n| Run | Width/head/MLP | Parameters | Best BPB | Final BPB | Best update | Training s | Validation s | GPU MiB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in widths['resources']:text+=f"|{r['experiment']}|{r['width']}/{r['head_dimension']}/{r['MLP_hidden']}|{r['parameters']:,}|{r['best_bpb']:.8f}|{r['final_bpb']:.8f}|{r['best_step']}|{r['training_seconds']:.3f}|{r['validation_seconds']:.3f}|{r['peak_gpu_MiB']:.3f}|\n"
    text+='\nE14 minus E01 (negative likelihood/BPB delta favors E14):\n\n| Metric | Absolute delta | Relative % |\n|---|---:|---:|\n'
    for r in pair['summary_comparison']:text+=f"|{r['metric']}|{r['delta_E14_minus_E01']:.8f}|{r['relative_percent']:.5f}|\n"
    text+=f'''
## Learning dynamics: observations and interpretation boundaries

Observed3k→5k validation loss change (improvement positive):
{s['loss_improvement_3000_to_5000']:.8f}. Recorded3k–5k fitted slope:
{s['late_validation_slope_per_1000_steps']:.8f}nats/1000updates.
Best checkpoint is{s['best_validation']['step']}; final minus best loss is
{s['final_validation']['loss']-s['best_validation']['loss']:.8f}.
These scheduled observations do not establish an asymptotic plateau, significant
overfitting or a general width effect. Same-step generalization gaps compare a
dropout-on sampled training batch with dropout-off full validation and are noisy.
Training averages all positions of sampled windows, including short conditioning
near each window start; sliding validation scores each target with up to128prior
characters. Thus loss gaps also reflect these conditioning differences.
No per-layer gradient time series or FLOP measurement was recorded.

Aggregate actual pre-clipping gradient statistics:

```json
{json.dumps(pair['gradient_statistics'],indent=2)}
```

## Complete genuine generations

All100 best/final protocol slots are preserved byte-for-byte in generation.json
and generation.md: five fixed prompts, greedy(seed42), temperatures0.7/1.0/1.2
with seeds42/123/456,256 new characters. Best/final identical checkpoint:
{duplicate}; duplicate slots are explicitly the same weights when true.
No output was rewritten. Five predetermined best greedy examples follow; all
sampled failures remain in the complete grid. Repetition fractions describe
repeated8-character windows and are not coherence or significance scores.

'''
    for r in grid['samples']:
        if r['checkpoint_policy']=='best' and r['greedy']:text+=f"### Prompt {r['prompt']!r}\n\nCheckpoint{r['checkpoint']}; seed42; greedy;256new tokens.\n\n```text\n{r['text']}\n```\n\n"
    text+='Aggregate fixed-protocol repetition measurements:\n\n```json\n'+json.dumps(quality['groups'],indent=2)+'\n```\n'
    text+='\nTraining phrase overlap is checked on ten predetermined outputs; ordinary phrases repeat and longest exact overlap is an indicator, not proof of memorization. See checkpoint_diagnostics.json. No test text or external model was used to select or improve outputs.\n'
    text+=f'''
## Durable artifacts and independent acceptance

Archive size{manifest['archive_bytes']:,}bytes.
SHA256:`{manifest['archive_sha256']}`.
All{len(manifest['files'])}exact archive entries passed independent CRC/path-safety/
SHA256 checks. All20 checkpoint counters/history/optimizer/scheduler states and
21 immutable snapshot manifests were verified. Native snapshots remain in saved
outputs; the compact archive holds every distinct resumable checkpoint and logs.
Best/final CPU inference passed. Fresh qualified CUDA restoration preserved exact
model, optimizer, scheduler, scaler, Python/NumPy/torchCPU/CUDA RNG and sampling/
history state at5000→5000, zero additional updates. CPU/CUDA recovery generation
matched. Receipt files, hashes and retrieval locations accompany this report.
The detailed report/comparisons are published as lightweight Git evidence;
large checkpoints, full journals and datasets remain outside Git.

## Failures, tests, limitations and next action

The authoritative5000-step journal records no research failure/resume/replay.
Development and retrieval/API/queue failures, if any, belong to the operational
record rather than altered research history. Test commands and CI evidence are
in evidence/final_checks.json and operations.md; CUDA success refers to actual
executed cloud receipts, not CPU tests.

One seed gives no variance/SD or statistical significance. Width, capacity, head
dimension, MLP width and initialization change together. Equal target counts do
not imply equal FLOPs/runtime; separate-session T4 timings include noise. No test
selection, additional seeds or unapproved10k extension was performed. Review the
actual quality/cost tradeoff and generation failures, then approve matched seeds
before claiming robust architecture superiority. Any10k continuation requires
separate approval and exact complete saved state; it has not been started.

See interpretation.md for evidence-based qualitative observations and budget
recommendation; operations.md for commands and platform incidents, and
artifact_retrieval.md for immutable saved versions/checkpoint locations.
'''
    text+='\n## Genuine figures\n\n'
    for directory in ['figures','comparison','width_comparison']:
        for p in sorted((out/directory).glob('*.png')):text+=f"![{p.stem}]({directory}/{p.name})\n\n"
    (out/'E14_seed42_analysis.md').write_text(prose_spacing(text).rstrip()+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out),'run_id':s['run_id'],'accepted_state_groups':gpu['exact_state']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','artifacts','comparison','widths','diagnostics','quality','manifest','recovery','platform','development','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['notebook','recovery-url']:p.add_argument('--'+name,required=True)
    report(p.parse_args())
