"""Render verified width-144 research evidence; no training or synthetic results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    source=a.artifacts
    s=read(source/'analysis/summary.json'); c=read(source/'completion.json')
    compared=read(a.comparison/'comparison.json'); d=read(a.diagnostics)
    gpu=read(a.recovery); local=read(source/'independent_recovery.json'); m=read(a.manifest)
    env=read(source/'environment.json'); hw=read(source/'hardware.json'); io=read(source/'io_timing.json')
    g=read(source/'analysis/generation.json'); audit=read(source/'width_audit.json')
    preflight=read(source/'preflight.json'); smoke=read(source/'CUDA_smoke_state_restore.json')
    quality=read(a.quality); timing=read(a.platform); review=read(a.review)
    assert s['parameters']==1021824 and s['completed_optimizer_updates']==5000
    assert c['validation_steps']==list(range(250,5001,250)) and len(read(source/'snapshot_index.json'))==21
    assert audit['status']=='passed' and audit['device']=='cuda' and len(audit['checks'])==20
    assert preflight['model']['d_model']==144 and preflight['accumulation']==8
    assert preflight['sampled_windows_per_update']==32 and preflight['predicted_positions_per_update']==4096
    assert all(smoke['exact_state'].values()) and smoke['additional_optimizer_updates']==0 and smoke['device']=='cuda'
    assert compared['comparability_passed'] and compared['only_model_factor_difference']=='d_model'
    assert compared['same_final_sampling_RNG_and_byte_coverage'] and len(compared['sampling_audits'])==2
    assert d['journal_steps']==5000 and d['failures']==d['resumes']==0
    assert gpu['status']=='fresh_saved_output_CUDA_state_restore_passed_no_updates'
    assert set(gpu['exact_state'])=={'model','optimizer','scheduler','scaler','rng','training_state'}
    assert gpu['run_id']==s['run_id'] and gpu['trained_source_commit']==s['source_commit']
    assert gpu['environment']['code_hash']==s['code_hash'] and not gpu['environment']['git_dirty']
    assert gpu['optimizer_updates_before']==gpu['optimizer_updates_after']==5000
    assert gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    assert local['verified_file_count']==len(m['files']) and local['archive_sha256']==m['archive_sha256']
    assert len(g['samples'])==100 and g['protocol']==read(a.root/'configs/generation_protocol.json')
    assert quality['generation_file_sha256']['E13']==hashlib.sha256((source/'analysis/generation.json').read_bytes()).hexdigest()
    out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'evidence').mkdir()
    shutil.copytree(a.comparison,out/'comparison');shutil.copytree(source/'analysis/figures',out/'figures')
    for name in ['summary.json','validation.csv','generation.json','generation.md']:
        shutil.copyfile(source/'analysis'/name,out/name)
    shutil.copyfile(a.quality,out/'generation_quality.json')
    receipts={'artifact_manifest.json':a.manifest,'checkpoint_diagnostics.json':a.diagnostics,
              'cuda_recovery.json':a.recovery,'platform_timing.json':a.platform,'execution_review.json':a.review}
    for name in ['completion','environment','hardware','dataset_validation','width_audit','preflight',
                 'CUDA_smoke_state_restore','storage_preflight','full_validation_preflight','execution_command',
                 'io_timing','wall_time','snapshot_index','independent_recovery']:
        receipts[name+'.json']=source/(name+'.json')
    for name,path in receipts.items():shutil.copyfile(path,out/'evidence'/name)
    observations={'training_updates_per_second':5000/s['training_seconds'],
                  'targets_per_training_second':s['training_tokens']/s['training_seconds'],
                  'checkpoint_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'),
                  'export_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='export'),
                  'journal_seconds':io['journal_seconds'],'generation_seconds':g['seconds'],
                  'archive_seconds':m['archive_seconds'],'pipeline_seconds':timing['pipeline_wall_seconds']}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf-8')
    text=f'''# E13 seed42: width-144 research experiment

## Scope, objective and hypothesis

One explicitly authorized E13 seed42 run completed5000 authoritative optimizer
updates and20full sliding validations. No additional configurations, seeds,
hybrids or extensions were launched. E11 acceptance passed before E13 execution:
the original recovery restored all state at5000→5000zeroextra, and PR15 was
merged atb37c2d82bfee067c28d06c2b3c8c0649774a6388. Earlier results and the
collaborator's E12 work are preserved. Study-wide approval remains separately gated.

Question: how does reducing model width192→144 affect prediction, optimization,
resource use and generated text? The smaller model might retain useful quality
with lower memory/compute, or lose modeling capacity. No direction was assumed.

## Architecture and numerical correctness

Only d_model changes. Four preLN blocks, six heads, context128, MLP expansion4,
GELU, sinusoidal positions and dropout0.1 remain fixed. Width144 mechanically
derives head dimension24 and MLP144→576→144. Attention output144→144,
all LayerNorm dimensions144, sinusoidal buffer128×144, vocabulary65,
embedding65×144 and LMhead144→65. No RoPE/learned/ReLU/BPE factor was added.
Trainable parameters **1,021,824**, **782,976 fewer** than E01
({782976/1804800*100:.6f}% reduction). MLP parameters166,608/block.

Twenty CPU/CUDA numerical checks verify the sole override/defaults, repeat-seed
initialization, independent blocks, all derived projections/counts, manual
linear/GELU/preLN residual references, Q/K/V shape(batch,6,128,24), independent
sqrt24/causal softmax/value aggregation, sinusoidal-only positions, shifted
next-token CE, finite nonzero parameter gradients/all blocks, future invariance,
dropout train/eval, context128 generation crop and overlength rejection.
Audit fixtures are development mathematics, not research measurements.

Matching seeds do not imply identical initial tensors across widths. Tensor
shapes, initialization fan-in scales, RNG consumption and sinusoidal frequency
spacing change mechanically. The initialization algorithm itself is unchanged.
Width changes capacity, head dimension, MLP width and embedding/output dimensions
together; this does not isolate any one of these consequences independently.

## Source, environment and dataset provenance

- Reviewed synchronized main:b37c2d82bfee067c28d06c2b3c8c0649774a6388.
- Clean trained source:`{s['source_commit']}`.
- Unchanged scientific hash:`{s['code_hash']}`.
- Run:`{s['run_id']}`; configuration hash:`{s['config_hash']}`.
- [Private research saved version1](https://www.kaggle.com/code/{hw['kernel_id']}).
- [Fresh zero-update CUDA recovery version1](https://www.kaggle.com/code/{a.recovery_kernel}).
- GPU:{env['gpu']}, device0, total memory{hw['devices'][0]['total_bytes']:,}bytes;
  {hw['gpu_count']} devices attached, one used for research training.
- Python{env['python']}, torch{env['torch']}, CUDA{env['cuda']}, NumPy1.26.4;
  OS:{env['platform']}; qualified transitive freeze reproduced exactly except editable source.

Canonical source revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e.
Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
All source/tokenizer/partition/manifest fields matched; vocabulary65;
contiguous train/validation/test bytes1,003,854/55,770/55,770. Train-only fitting,
separate windows, no held-out training leakage, no test likelihood/selection.
Native Kaggle packages were untouched; no CPU fallback or uncommitted source.

## Training, evaluation and persistence preflight

Fresh seed42 initialization; AdamW LR0.0003constant, weight decay0.01, clipping1,
float32 CUDA, microbatch4, accumulation8,4096targets/update,32sampled windows/update,
5000updates. Full sliding validation every250,checkpoint cadence500 while
retaining every validation state. No AMP, early stopping or schedule change.
Independent real-data development smoke precedes full research reseeding.

Validation scores every one of55,514original suffix bytes after the common
256-byte exclusion, at context128/batch32/dropoutoff. Sliding crops reset local
sinusoidal positions. BPB=sumNLL/(ln2×scored bytes), character BPB=loss/ln2,
perplexity=exp(loss). These are related metrics. No capped research validation
or invented step-zero measurement; token perplexities are not compared with BPE.

Preflight passed real CUDA backward, full validation, checkpoint reload,
immutable snapshot export/restore, actual optimizer/scheduler/scaler/Python-NumPy-
torchCPU/CUDA-RNG/sampler/history roundtrip with zeroextra development updates,
restored inference, ZIPCRC/safe paths/embedded checkpoint SHA256, storage budget,
Agg plotting and generation. Actual checkpoint sizes drive storage planning.
E11's published and independently recovered account workflow was verified before
E13; E13 publication/retrieval is verified below, not inferred from local paths.

## Tests and recorded results

CPU suite{review['windows_suite']['passed']}passed/{review['windows_suite']['skipped']}CUDA-only skip;
focused{review['focused_suite']['passed']}passed;
actual Kaggle{review['actual_Kaggle_suite']['passed']}passed in{review['actual_Kaggle_suite']['seconds']:.2f}s.
CPU/CUDA20-check width audits, real-data smoke and all recovered checkpoint
counter/BPB/sampler/archive checks passed. Actual commands/CI are in evidence.

| Quantity | Recorded value |
|---|---:|
| First training loss/update1 |{s['initial_recorded_training']['loss']:.8f}|
| First validation loss/update250 |{s['initial_recorded_validation']['loss']:.8f}|
| Final training loss |{s['final_training_loss']:.8f}|
| Final100-update training mean |{s['final_training_trailing_100_loss']:.8f}|
| Best checkpoint |{s['best_validation']['step']}|
| Best loss/BPB |{s['best_validation']['loss']:.8f}/{s['best_validation']['bpb']:.8f}|
| Final loss/BPB |{s['final_validation']['loss']:.8f}/{s['final_validation']['bpb']:.8f}|
| Final perplexity |{s['final_validation']['perplexity']:.8f}|
| Predicted targets/source-byte exposure |{s['training_tokens']:,}/{s['training_source_bytes']:,}|
| Unique target-byte coverage |{s['unique_target_bytes']:,}|
| Training-only seconds |{s['training_seconds']:.3f}|
| Full-validation seconds |{s['validation_seconds']:.3f}|
| Measured clone-to-archive seconds |{observations['pipeline_seconds']:.3f}|
| Updates/training second |{observations['training_updates_per_second']:.5f}|
| Targets/training second |{observations['targets_per_training_second']:.3f}|
| Peak allocated GPU MiB |{s['peak_gpu_bytes']/2**20:.3f}|

Training time excludes evaluation/IO. Pipeline timing excludes queue/publication/
network retrieval/fresh recovery. Peak allocation excludes reserved/driver memory.
Repeated target-byte exposure is not input traffic, unique examples, epochs or FLOPs.

```json
{json.dumps(observations,indent=2)}
```

## All twenty matched E01/E13 validations

| Update | E01 loss | E13 loss | Δloss | E01 BPB | E13 BPB |
|---|---:|---:|---:|---:|---:|
'''
    for row in compared['matched_validation']:
        text+=f"|{row['step']}|{row['E01_loss']:.8f}|{row['E13_loss']:.8f}|{row['delta_loss']:.8f}|{row['E01_bpb']:.8f}|{row['E13_bpb']:.8f}|\n"
    text+='\n## Predictive quality and measured resource tradeoffs\n\n| Metric | E01 | E13 | ΔE13−E01 | Relative % |\n|---|---:|---:|---:|---:|\n'
    for row in compared['summary_comparison']:
        text+=f"|{row['metric']}|{row['E01']:.8f}|{row['E13']:.8f}|{row['delta_E13_minus_E01']:.8f}|{row['relative_percent']:.3f}|\n"
    text+='\nLower loss/BPB is better; fewer parameters alone do not establish superior quality. Equal targets are not equal FLOPs. Separate T4 sessions/no repeated timing confound speed comparisons; BPB divided by parameter count is not an intrinsic quality score.\n'
    text+='\n```json\n'+json.dumps(compared['sampling_audits'],indent=2)+'\n```\n\nBoth private samplers were independently replayed and final RNG/target unions match E01. Model/global initialization/dropout RNG are not assumed equal across widths.\n'
    for folder in ['comparison','figures']:
        for p in sorted((out/folder).glob('*.png')):text+=f'\n![Measured E13 evidence: {p.stem}]({folder}/{p.name})\n'
    text+=f"\n## Convergence and gradient observations\n\nObserved3k–5k loss decrease{s['loss_improvement_3000_to_5000']:.8f}; late recorded-point slope{s['late_validation_slope_per_1000_steps']:.8f} per1000updates; final-minus-best{s['final_validation']['loss']-s['best_validation']['loss']:.8f}. These are descriptors, not forecasts or interpolated observations.\n\n```json\n{json.dumps(compared['gradient_statistics'],indent=2)}\n```\n\nAggregate pre-clipping norms establish neither longitudinal per-layer behavior nor vanishing/exploding gradients. Training/dropout-on sampled windows and full sliding/dropout-off validation differ; their gap is not a direct memorization rate. Detailed interpretation is in interpretation.md.\n"
    text+='\n## Complete genuine model generation\n\n[All100 unedited outputs](generation.md) preserve five fixed prompts, greedy, temperatures0.7/1/1.2, seeds42/123/456,256new tokens, best/final and128-token crops. Five predetermined best greedy examples follow, including poor outputs. No external model produced or improved them.\n'
    for row in g['samples']:
        if row['checkpoint_policy']=='best' and row['greedy']:
            text+=f"\n### Prompt {row['prompt']!r}\n\nCheckpoint{row['checkpoint']};greedy;seed{row['seed']};256new tokens.\n\n```text\n{row['text']}\n```\n"
    if c['best_checkpoint']==c['final_checkpoint']:
        text+='\nBest/final are identical weights; the separately labeled grids duplicate deterministic settings, not100independent draws.\n'
    text+='\nAll100matched generation settings/repetition statistics are in generation_quality.json. Repetition is descriptive, not a coherence score. Exact train-phrase overlap in checkpoint_diagnostics.json is not conclusive memorization proof. Actual word/grammar/dialogue/prompt/repetition failures are discussed in interpretation.md without cherry-picking.\n'
    text+=f"\n## Durable publication and independently verified recovery\n\nArchive:{m['archive_bytes']:,}bytes; SHA256`{m['archive_sha256']}`. Independently retrieved archive passed safe paths,ZIPCRC,exactinventory and all{len(m['files'])}filehashes; best/final CPUinference passed. All20resumable checkpoint counters/histories verified. Native wrapper verified21immutable exports before publication; independently retrieved snapshotmanifest parity is documented separately. Fresh qualified CUDA restoration restored all model/optimizer/scheduler/scaler/Python-NumPy-torchCPU/CUDA-RNG/sampler/history fields exactly at5000→5000 withzeroextra. Generation samples match recovered CPUoutputs. Large archives/weights/fulljournals remain outsideGit; current retrieval root:`{source}`. See artifact_retrieval.md and artifact_manifest.json.\n\nJournal:{d['failures']}research failure events,{d['resumes']}resume events. In the absence of interruptions, authoritative and known physical research updates both5000; independent development smoke is excluded. Any transport/API/setup failures remain in operations.md. No superseded/replayed research trajectory is fabricated or merged.\n\n## Scientific conclusions and future width comparison\n\n[Interpretation and recommendations](interpretation.md) distinguish hypotheses, observations and conclusions. One seed/dataset gives no SD/significance/equivalence estimate. Width changes capacity/head dimension/MLP width/initialization together; no isolated causality is claimed. The framework contains genuineE13/E01 results and E14=null/NOT_AUTHORIZED_NOT_RUN. NoE14/extra seed/extension or untested hybrid was executed.\n"
    (out/'E13_seed42_analysis.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','artifacts','comparison','diagnostics','recovery','manifest','quality','platform','review','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--recovery-kernel',required=True)
    report(parser.parse_args())
