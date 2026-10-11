"""Render a reproducible pilot report from independently retrieved measured artifacts."""
import argparse
import json
from pathlib import Path
import shutil
from nano_lm.checkpoint import read_checkpoint


def report(artifacts,platform,recovery,out):
    out.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((artifacts/name).read_text())
    c=read('completion.json');s=read('analysis/summary.json');io=read('io_timing.json');hardware=read('hardware.json');env=read('environment.json')
    local=read('independent_recovery.json');timing=json.loads(platform.read_text());gpu=json.loads(recovery.read_text())
    assert gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    generation=read('analysis/generation.json');history=read('analysis/measured_history.json')['training'];validation=[r for r in history if 'validation' in r]
    checkpoint=artifacts/c['final_checkpoint'];payload,_=read_checkpoint(checkpoint)
    figures=out/'figures';figures.mkdir()
    for p in (artifacts/'analysis/figures').glob('*.png'):shutil.copyfile(p,figures/p.name)
    for name in ['summary.json','generation.json','generation.md','validation.csv']:shutil.copyfile(artifacts/'analysis'/name,out/name)
    cp=sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint');export=sum(r['seconds'] for r in io['records'] if r['operation']=='export')
    observed={'completed_updates':s['completed_optimizer_updates'],'validation_points':len(validation),'minimum_loss_step':s['best_validation']['step'],'minimum_loss':s['best_validation']['loss'],'final_loss':s['final_validation']['loss'],'late_slope_per_1000':s['late_validation_slope_per_1000_steps'],'loss_improvement_3000_to_5000':s['loss_improvement_3000_to_5000'],'checkpoint_seconds':cp,'snapshot_export_seconds':export,'journal_seconds':io['journal_seconds'],'pipeline_wall_seconds':timing['pipeline_wall_seconds'],'gradient_clipping_fraction':sum(r['gradient_norm']>1 for r in history)/len(history),'learning_rates':sorted(set(r['lr'] for r in history))}
    (out/'pilot_observations.json').write_text(json.dumps(observed,indent=2),encoding='utf-8')
    text=f'''# E01 seed42 baseline training pilot — 2026-10-12

This is one genuine single-seed baseline pilot, not a completed three-seed result
or 14-configuration study. It executed exactly **5000 optimizer updates**.
Study-wide NO-GO remains because the second independent Kaggle account is unavailable.
The user explicitly authorized this one-account pilot as a scoped exception.

## Objective and hypotheses

Assess whether 5000 updates is a reasonable initial screening budget for the
locked Tiny Shakespeare ablations, and establish real learning/storage costs.
Before execution, improvement, plateau and deterioration were all possible;
overfitting and double descent were not assumed. A single baseline curve cannot
establish convergence of every deeper/wider/tokenizer/context variant.

## Exact source and reproducibility

- Trained source: `{s['source_commit']}`; clean committed checkout.
- Canonical main synchronized before orchestration: `36ffee3926f12f58582b3502f725aecc529a1f8d`.
- Code hash: `{s['code_hash']}`; unchanged qualified model/training/evaluation code.
- Config hash: `{s['config_hash']}`.
- Run identity: `{s['run_id']}`.
- Saved private Kaggle notebook version1: https://www.kaggle.com/code/{hardware['kernel_id']}
- GPU: {env['gpu']}; GPU0 only, {hardware['gpu_count']} devices allocated.
- Python {env['python']}, PyTorch {env['torch']}, torch CUDA {env['cuda']}.
- Qualified resolved freeze reproduced exactly except the editable project SHA.
- Full metadata, hardware/driver output, command arguments and freeze are in the
  checksum-verified archive, not copied from uncommitted Windows source.

Canonical architecture: explicit decoder-only causal Transformer, character
vocabulary65, sinusoidal positions, width192, layers4, heads6, context128,
MLP expansion4, GELU, dropout0.1. Parameters: **{s['parameters']:,}**.
No pretrained weights or substitute architecture were used.

Optimizer AdamW, constant LR0.0003, weight decay0.01, clipping norm1.0,
seed42, float32 CUDA. Microbatch4, accumulation8, context128 gives exactly
4096 predicted tokens/update. No early stopping, AMP, schedule change, architecture
change or automatic 10k extension. A separate one-update development preflight
verified backward/save/reload; the engine reseeded for the independent research run.

## Data and evaluation

Pinned Karpathy char-rnn revision `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`.
Raw SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Character tokenizer SHA256 `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`.
Train/validation/test contiguous source bytes:1003854/55770/55770. All reference
source/split/tokenizer/manifest fields matched. Train-only fitting and windows;
no test likelihood was evaluated or used for selection.

Full sliding validation every250 updates, batch32, context128, on the complete
55770-byte validation text. First256 original bytes condition predictions and
are excluded; **55514 bytes/tokens scored** once per evaluation. Conditional BPB
is summed NLL / (ln(2)*scored bytes), not whole-file compression entropy. Perplexity
is exp(mean token NLL). There is no invented step-zero evaluation.

Training loss measures sampled batches with dropout active; validation uses eval
mode without dropout. Same-step generalization gaps are noisy diagnostics, not
equal-protocol measurements of train/validation corpus likelihood. The last100
training losses are averaged only as a clearly labeled descriptive smoother.
Context truncation resets local position indices, as in the canonical evaluator.

## Actual measured results

| Quantity | Recorded value |
|---|---:|
| Initial training loss, update1 | {s['initial_recorded_training']['loss']:.8f} |
| First validation loss, update250 | {s['initial_recorded_validation']['loss']:.8f} |
| Final same-step training loss | {s['final_training_loss']:.8f} |
| Final100-step trailing training mean | {s['final_training_trailing_100_loss']:.8f} |
| Final validation loss, update5000 | {s['final_validation']['loss']:.8f} |
| Best validation loss | {s['best_validation']['loss']:.8f} |
| Best validation update | {s['best_validation']['step']} |
| Final conditional BPB | {s['final_validation']['bpb']:.8f} |
| Best conditional BPB | {s['best_validation']['bpb']:.8f} |
| Final validation perplexity | {s['final_validation']['perplexity']:.8f} |
| Processed predicted tokens/source bytes | {s['training_tokens']:,} / {s['training_source_bytes']:,} |
| Unique target source bytes | {s['unique_target_bytes']:,} |
| Peak PyTorch allocated GPU memory, MiB | {s['peak_gpu_bytes']/2**20:.3f} |

Repeated target bytes count exposure, not independent examples or completed
nonoverlapping epochs. The first training byte cannot be a next-token target
without a preceding token. GPU allocated-memory peaks exclude driver overhead
and reserved memory; they are not whole-device memory use.

| Update | Full validation loss | Conditional BPB | Perplexity |
|---|---:|---:|---:|
'''
    for r in validation:
        v=r['validation'];text+=f"| {r['step']} | {v['loss']:.8f} | {v['bpb']:.8f} | {v['perplexity']:.8f} |\n"
    text+=f'''
## Measured time and overhead

| Scope | Seconds |
|---|---:|
| Training only, engine measurement | {s['training_seconds']:.3f} |
| Full validation total,20 actual evaluations | {s['validation_seconds']:.3f} |
| Actual checkpoint function calls | {cp:.3f} |
| Actual verified snapshot-export calls | {export:.3f} |
| Journal append/flush/fsync calls | {io['journal_seconds']:.3f} |
| Training CLI wall time | {io['cli_wall_seconds']:.3f} |
| Original notebook pipeline wall, clone through postprocessing failure | {timing['pipeline_wall_seconds']:.3f} |

Original pipeline wall uses actual Kaggle elapsed-log timestamps and includes
isolated environment setup, data/preflight, training and the subsequent plotting
import failure. Postprocessing/recovery pipeline time is reported separately;
the original notebook did not produce its planned archive. Queue before execution,
final platform conversion/output publication and network retrieval are separate. IO hook timings
forwarded original calls without changing their arguments/returns or training
math. CLI wall minus measured scopes includes setup, loop bookkeeping and other
unattributed overhead; it is not relabeled checkpoint IO. Individual validation
durations were not separately logged; their cumulative engine duration is genuine.

## Figures from genuine history

Lines connect observed validation points, not invented intermediate evaluations.
Training smoothing is the declared100-step trailing mean. PDFs remain in the
verified full artifact; the PNG copies below are provided for the research report.

'''
    for name in ['training_loss','validation_loss','combined_loss','validation_bpb','generalization_gap','throughput','training_time','gradient_norm','learning_rate']:
        text+=f"![{name.replace('_',' ')}](figures/{name}.png)\n\n"
    text+=f'''## Convergence observations

Measured validation improvement from3000 to5000: **{s['loss_improvement_3000_to_5000']:.8f} nats/token**.
Descriptive linear slope over actual3000–5000 validation points:
**{s['late_validation_slope_per_1000_steps']:.8f} nats/token per1000 updates**.
Best observed checkpoint is step{s['best_validation']['step']}; final validation
is {s['final_validation']['loss']-s['best_validation']['loss']:.8f} above that minimum.
These descriptors have no single-seed significance interpretation. A negative
slope means the fitted late trend improved, not that every interval improved.
Gradient norms exceeded clipping threshold on {observed['gradient_clipping_fraction']:.2%}
of updates; the norms are measured before clipping. LR remained constant.

Interpretation, sample-quality review and the budget recommendation follow the
descriptive measurements after inspection of the full curve and unedited generations. Neither plateau,
overfitting nor double descent is inferred solely from a local fluctuation.

## Genuine generations and limitations

[Complete unedited generation grid](generation.md) and generation.json preserve
all5 fixed prompts, greedy decoding and temperatures0.7/1.0/1.2 with sampling
seeds42/123/456,256 new tokens, from both best and final checkpoints. GPU samples
come directly from this project's generate function. No external model produced
or corrected them. Five predetermined best-checkpoint greedy examples follow;
they are selected by protocol, not favorable quality. Character spelling,
syntactic coherence, dialogue structure, repetitions and training-text overlaps
are diagnostics; fluent-looking samples do not establish semantic understanding.

'''
    for sample in generation['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:
            text+=f"### Prompt {sample['prompt']!r}\n\nCheckpoint `{sample['checkpoint']}`, greedy, seed42,256 new tokens.\n\n```text\n{sample['text']}\n```\n\n"
    text+=f'''## Artifact preservation and recovery

Final checkpoint: `{c['final_checkpoint']}`.
Best checkpoint: `{c['best_checkpoint']}`.
All20 validation/checkpoint states, including1k/2k/3k/4k/5k, remain in the complete
run and published notebook output. Optimizer/scheduler/scaler/RNG/sampling state,
tokenizer, dataset manifest, immutable metadata,5000-row append-only history,
20-point validation history, snapshots, plots and generations are preserved.

Archive SHA256: `{local['archive_sha256']}`; bytes:{local['archive_bytes']:,}.
All **{local['verified_file_count']} archive files** were independently retrieved,
CRC/hash/path-checked. Best/final checkpoints loaded and generated on Windows CPU.
Fresh Kaggle saved-output CUDA recovery passed exact model, optimizer, scheduler,
scaler, RNG and training-state roundtrip; **zero additional updates** executed.
The actual10k extension has not begun. Recovery source `{gpu['source_commit']}`
adds only post-run utilities; canonical code hash and package freeze matched.

The archive contains checkpoints once; all immutable snapshot copies additionally
remain in saved notebook output. Publication/download are independently checked,
not inferred from successful /kaggle/working exports. Notebook batch execution
automatically ends to publish outputs; no manual runtime cancellation was issued.
Full file manifest/retrieval commands are recorded alongside this report.

## Scientific limitations and next stage

One seed provides no across-seed mean/SD or statistical significance. A baseline
curve cannot prove equal convergence budgets for deeper, wider, other-context or
BPE variants. Fixed4096-token budgets do not equal fixed FLOPs or equal unique
text exposure. Validation informs checkpoint selection; test data remains reserved.
Character-model BPB is tokenizer-specific conditional validation evidence; it is
not a completed cross-tokenizer comparison. The second environment is unavailable.
No other research run or automatic extension is authorized by this report.
'''
    (out/'E01_seed42_analysis.md').write_text(text,encoding='utf-8')
    print(out/'E01_seed42_analysis.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--platform',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.artifacts,a.platform,a.recovery,a.output)
