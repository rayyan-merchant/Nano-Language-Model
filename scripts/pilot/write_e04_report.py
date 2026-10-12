"""Render genuine E04 evidence; qualitative interpretation is reviewed separately."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from nano_lm.checkpoint import read_checkpoint


def report(a):
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    artifact=a.artifacts;out=a.output;out.mkdir(parents=True,exist_ok=False)
    completion=read(artifact/'completion.json');s=read(artifact/'analysis/summary.json')
    m=read(a.manifest);gpu=read(a.recovery);local=read(artifact/'independent_recovery.json')
    comp=read(a.comparison/'comparison.json');platform=read(a.platform)
    d=read(a.diagnostics);io=read(artifact/'io_timing.json');env=read(artifact/'environment.json')
    hw=read(artifact/'hardware.json');generations=read(artifact/'analysis/generation.json')
    assert s['run_id'].startswith('E04_seed42_') and s['completed_optimizer_updates']==5000
    assert s['parameters']==1829376 and comp['comparability_passed'] and local['optimizer_updates']==5000
    assert gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    assert len(generations['samples'])==100 and generations['protocol']==read(a.root/'configs/generation_protocol.json')
    p,_=read_checkpoint(artifact/completion['final_checkpoint']);history=p['training_state']['history']
    vals=[r for r in history if 'validation' in r]
    assert [r['step'] for r in vals]==list(range(250,5001,250)) and len(history)==5000
    for name in ['summary.json','generation.json','generation.md','validation.csv']:shutil.copyfile(artifact/'analysis'/name,out/name)
    shutil.copytree(a.comparison,out/'comparison');(out/'figures').mkdir();e=out/'evidence';e.mkdir()
    for f in (artifact/'analysis/figures').glob('*.png'):shutil.copyfile(f,out/'figures'/f.name)
    receipts={'artifact_manifest.json':a.manifest,'cuda_recovery.json':a.recovery,'platform_timing.json':a.platform,
              'checkpoint_diagnostics.json':a.diagnostics,'execution_review.json':a.review}
    for name in ['completion','dataset_validation','environment','execution_command','hardware','io_timing','preflight','learned_audit','snapshot_index','wall_time','independent_recovery']:
        receipts[name+'.json']=artifact/(name+'.json')
    for name,path in receipts.items():shutil.copyfile(path,e/name)
    (e/'receipt_checksums.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in e.iterdir()},indent=2)+'\n')
    cp=sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint')
    ex=sum(r['seconds'] for r in io['records'] if r['operation']=='export')
    clipped=sum(r['gradient_norm']>1 for r in history)
    observations={'updates':5000,'validation_count':20,'clipped_updates':clipped,'clipping_fraction':clipped/5000,
                  'learning_rates':sorted({r['lr'] for r in history}),'checkpoint_seconds':cp,'snapshot_export_seconds':ex,
                  'all_generation_paired_best_final_equal':all(generations['samples'][i]['text']==generations['samples'][i+50]['text'] for i in range(50))}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    text=f'''# E04 seed42 learned absolute positional embeddings — 2026-10-12

One explicitly authorized research run, exactly **5000 completed optimizer
updates**, all twenty full validations. No other configuration/seed/10k extension.
E01/E02/E03 historical evidence is preserved. This single-seed observation is not
a completed three-seed study or a statistical significance result.

## Research objective and hypothesis

Measure how trainable absolute position vectors change next-token prediction,
learning dynamics, runtime/memory and fixed-prompt generation relative to additive
sinusoidal positions and RoPE. **Hypothesis:** training-distribution-specific
position vectors might help or hurt prediction. No direction was assumed.

## Learned-position implementation and audit

Canonical GPT adds token embeddings to `nn.Embedding(128,192)` indexed0..L-1
before unchanged dropout and four pre-LayerNorm blocks. Neither sinusoidal addition
nor Q/K rotary transformation is active. Each head remains32 dimensions; unchanged
causal attention prevents future leakage. All position rows received finite,
nonzero gradients in the full-context numerical audit. Thirteen independent
checks passed on CPU and actual CUDA, including intercepted embedding addition,
poisoned unused sinusoidal buffer, prohibited RoPE, mask/shapes, manual CE,
future invariance and cropped generation indices.

E04 has **1829376 parameters**, +24576 (+{24576/1804800*100:.4f}%) versus
E01/E03. Shared parameter shapes are unchanged. The extra embedding initializes
before later layers and consumes seeded RNG draws: seed42 token embeddings match
E01, but later initial weights differ. The established implementation/initialization
was not altered. Added capacity and this initialization difference limit causal
attribution solely to the representation. Fresh E04 weights were used, never
E01/E03 fine-tuning. Local context indices reset after sliding/cropping exactly as
in training/evaluation/generation; no extrapolation beyond128 was tested.

## Exact reproducible setup

- Reviewed starting main:7b4d922c50e9f1663fff8f02829eb95545b9eb7f.
- Clean trained source:`{s['source_commit']}`.
- Scientific code hash:`{s['code_hash']}`; equivalent to E01–E03.
- Configuration hash:`{s['config_hash']}`.
- Run ID:`{s['run_id']}`.
- Original private published Kaggle v1:https://www.kaggle.com/code/{hw['kernel_id']}
- Fresh GPU recovery v1:https://www.kaggle.com/code/{a.recovery_kernel}
- Python{env['python']},PyTorch{env['torch']},torch CUDA{env['cuda']};Tesla T4 GPU0.
- NumPy1.26.4 and entire qualified dependency freeze reproduced; native runtime untouched.

Character65,width192,layers4,heads6,context128,MLP4,GELU,dropout.1,learned positions.
AdamW,constant LR.0003,weight decay.01,clip1,float32 CUDA,seed42,micro4/acc8,
4096 targets/update,validation250,checkpoint500 with existing validation saves.
Full command/environment/driver and source metadata are in evidence and archive.
Preflight development training/export/Agg figures/generation was labeled smoke;
the research engine reseeded independently. No development metric is mixed here.

## Dataset, separation and evaluation

Raw SHA256:`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Character tokenizer SHA256:`ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`.
Canonical acquisition revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e;
train/validation/test1003854/55770/55770bytes. All audited reference hashes matched.
Partitions precede token fitting/windows; no boundary crossing or test likelihood.
Full sliding validation excludes first256 conditioning bytes and scores55514
target bytes once. Conditional BPB=sum NLL/(ln(2)*55514); for ASCII character
targets this equals loss/ln(2), independently verified at every point. It is
conditional likelihood, not whole-file compression entropy. Validation uses
evaluation mode; sampled training has dropout. Generalization gaps are descriptive
and not equal-protocol corpus likelihood comparisons. No step-zero validation is invented.

## Genuine measured results

| Quantity | E04 recorded value |
|---|---:|
| Initial training loss,update1 | {s['initial_recorded_training']['loss']:.8f} |
| Initial validation loss,update250 | {s['initial_recorded_validation']['loss']:.8f} |
| Final training batch loss | {s['final_training_loss']:.8f} |
| Final100-update training mean | {s['final_training_trailing_100_loss']:.8f} |
| Best validation step | {s['best_validation']['step']} |
| Best validation loss | {s['best_validation']['loss']:.8f} |
| Best conditional BPB | {s['best_validation']['bpb']:.8f} |
| Final validation loss | {s['final_validation']['loss']:.8f} |
| Final conditional BPB | {s['final_validation']['bpb']:.8f} |
| Final within-character perplexity | {s['final_validation']['perplexity']:.8f} |
| Predicted token exposure | {history[-1]['training_tokens']} |
| Source-byte exposure including repeats | {history[-1]['training_bytes']} |
| Unique target bytes | {history[-1]['unique_target_bytes']} |
| Updates with preclip gradient norm>1 | {clipped} ({clipped/5000*100:.4f}%) |

| Update | Full validation loss | Conditional BPB | Perplexity |
|---:|---:|---:|---:|
'''
    for r in vals:text+=f"| {r['step']} | {r['validation']['loss']:.8f} | {r['validation']['bpb']:.8f} | {r['validation']['perplexity']:.8f} |\n"
    text+='\n## Three-way controlled character-position comparison\n\n| Metric | E01 | E03 | E04 |\n|---|---:|---:|---:|\n'
    for r in comp['summary_comparison']:text+=f"| {r['metric']} | {r['E01']:.8f} | {r['E03']:.8f} | {r['E04']:.8f} |\n"
    text+='\nAll twenty matching evaluations/deltas are in comparison/matched_validation.csv.\nFinal private sampling RNG and full target-byte coverage arrays matched all three.\nShared code/environment/settings are validated, not assumed from experiment names.\n'
    e02=comp['E02_contextual_summary']
    text+='\n## Overall study BPB context, single seed each\n\n| Configuration | Tokenizer | Parameters | Best BPB | Best step |\n|---|---|---:|---:|---:|\n'
    for eid in ['E01','E03','E04']:
        summary=s if eid=='E04' else read(a.root/f'report/pilots/{eid}_seed42/summary.json')
        text+=f"| {eid} | character | {summary['parameters']} | {summary['best_validation']['bpb']:.8f} | {summary['best_validation']['step']} |\n"
    if e02:text+=f"| E02 | BPE | {e02['parameters']} | {e02['best_validation']['bpb']:.8f} | {e02['best_validation']['step']} |\n"
    text+='\nE02 scores the same original suffix, but vocabulary1024, more parameters, longer\noriginal-byte contexts and49991931 training target bytes differ. Its best BPB is\ncontextual conditional comparison, not a controlled position ablation. Do not rank\nE02 token NLL or perplexity against characters. Fixed4096-token/update budgets\ndo not equal source exposure, FLOPs or compute. No cross-seed SD is estimated.\n'
    text+=f'''
## Timing and hardware costs

| Measured scope | Seconds |
|---|---:|
| Training only | {s['training_seconds']:.6f} |
| Full validation,20 evaluations | {s['validation_seconds']:.6f} |
| Checkpoint function calls | {cp:.6f} |
| Snapshot export function calls | {ex:.6f} |
| Append/flush/fsync journal calls | {io['journal_seconds']:.6f} |
| Canonical CLI wall | {io['cli_wall_seconds']:.6f} |
| Original notebook clone-through-archive pipeline | {platform['pipeline_wall_seconds']:.6f} |

Training-only{5000/s['training_seconds']:.6f}updates/sec,
{20480000/s['training_seconds']:.6f}targets/sec. Peak allocated memory
{s['peak_gpu_bytes']}bytes ({s['peak_gpu_bytes']/2**20:.6f}MiB), excluding reserved,
driver and whole-device overhead. Separate T4 sessions limit speed attribution.
Pipeline excludes pre-execution queue and final conversion/publication/retrieval;
unmeasured scopes are not inferred from timestamps. Full-validation time is
accumulated; individual evaluation durations were not separately logged.

## Convergence observations

Measured3k–5k improvement:{s['loss_improvement_3000_to_5000']:.8f}nats/token.
Descriptive late fitted slope:{s['late_validation_slope_per_1000_steps']:.8f}nats/token
per1000updates. Final loss minus best:{s['final_validation']['loss']-s['best_validation']['loss']:.8f}.
These describe scheduled observations, not unseen intermediate minima or
significance. [Reviewed interpretation and negative qualitative findings](interpretation.md)
discuss interval behavior, comparative learning and future budget recommendations.

## Figures from real history

Validation markers are actual observations; connecting lines are not extra
measurements. Training smoother is a declared100-update trailing mean.
'''
    for name in ['training_loss','validation_loss','combined_loss','validation_bpb','generalization_gap','throughput','training_time','gradient_norm','learning_rate']:
        text+=f'\n![{name}](figures/{name}.png)\n'
    for name in ['loss','bpb','training','throughput']:text+=f'\n![comparison {name}](comparison/comparison_{name}.png)\n'
    text+='\n## Genuine fixed-protocol generations\n\n[Complete unedited100-row grid](generation.md), generation.json: all five prompts,\ngreedy seed42 and temperatures.7/1/1.2,seeds42/123/456,256new tokens,best/final.\nNo external model edited outputs. Five preselected best greedy samples follow.\n'
    for sample in generations['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:
            text+=f"\n### Prompt {sample['prompt']!r}\n\nCheckpoint `{sample['checkpoint']}`, greedy seed42,256new tokens.\n\n```text\n{sample['text']}\n```\n"
    text+=f'''
## Durable publication, recovery and complete manifest

Final:`{completion['final_checkpoint']}`. Best:`{completion['best_checkpoint']}`.
Archive{m['archive_bytes']}bytes SHA256:`{m['archive_sha256']}`.
All{len(m['files'])}files passed independent CRC/safe-path/file-set/SHA verification.
All20 resumable checkpoints and21 immutable snapshots remain in saved outputs;
the compact archive stores checkpoint files once. Optimizer/scheduler/disabled
scaler/RNG/sampling/history/configuration/dataset/tokenizer/environment are retained.
CPU best/final inference passed after independent retrieval. Fresh compatible
CUDA restore exactly round-tripped all state fields at5000 with zero extra updates
and actual best/final generation. No10k continuation started.
[Complete per-file manifest](evidence/artifact_manifest.json) and checksum-preserved
receipts document actual publication and restore. Large artifacts stay outside GitHub.

| Checkpoint update | SHA256 |
|---:|---|
'''
    for name,h in sorted(m['files'].items()):
        if '/checkpoints/' in name and name.endswith('.pt'):text+=f"| {int(Path(name).stem.split('_')[-1])} | `{h}` |\n"
    text+='\n## Limitations\n\nOne seed has no variance/significance estimate. Added capacity and canonical RNG\ninitialization effects confound pure representation attribution. Fixed128 context\ndoes not test length extrapolation. Validation selects checkpoints; test remains\nreserved. Teacher-forced prediction and autoregressive text quality differ. Short\ntraining-phrase overlaps neither prove nor exclude memorization. Separate sessions\nare not repeated controlled speed benchmarks. Read interpretation.md for actual\nfailures, tests, retrieval paths and the evidence-backed next-step recommendation.\n'
    (out/'E04_seed42_analysis.md').write_text(text,encoding='utf8')
    print(out/'E04_seed42_analysis.md')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifacts','comparison','platform','recovery','manifest','diagnostics','review','root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-kernel',required=True);report(p.parse_args())
