"""Render E08 evidence and validated E01 comparison; interpretation is reviewed separately."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from nano_lm.checkpoint import read_checkpoint


def report(artifacts, comparison, platform, recovery, manifest, diagnostics, review, root, out, recovery_kernel):
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    c=read(artifacts/'completion.json');s=read(artifacts/'analysis/summary.json')
    io=read(artifacts/'io_timing.json');env=read(artifacts/'environment.json');hw=read(artifacts/'hardware.json')
    local=read(artifacts/'independent_recovery.json');gpu=read(recovery);m=read(manifest)
    compared=read(comparison/'comparison.json');d=read(diagnostics);t=read(platform)
    assert gpu['additional_optimizer_updates']==0 and all(gpu['exact_state'].values())
    assert local['optimizer_updates']==5000 and compared['comparability_passed']
    assert s['run_id'].startswith('E08_seed42_') and s['completed_optimizer_updates']==5000
    assert s['parameters']==2694528 and compared['only_model_factor_difference']=='n_layers'
    audit=read(artifacts/'depth_audit.json')
    assert audit['status']=='passed' and audit['device']=='cuda' and len(audit['checks'])==13
    payload,_=read_checkpoint(artifacts/c['final_checkpoint']);history=payload['training_state']['history']
    val=[r for r in history if 'validation' in r]
    assert [r['step'] for r in val]==list(range(250,5001,250))
    generations=read(artifacts/'analysis/generation.json');assert len(generations['samples'])==100
    assert generations['protocol']==read(root/'configs/generation_protocol.json')
    for name in ['summary.json','generation.json','generation.md','validation.csv']:shutil.copyfile(artifacts/'analysis'/name,out/name)
    shutil.copytree(comparison,out/'comparison')
    figures=out/'figures';figures.mkdir()
    for p in (artifacts/'analysis/figures').glob('*.png'):shutil.copyfile(p,figures/p.name)
    evidence=out/'evidence';evidence.mkdir()
    receipts={'artifact_manifest.json':manifest,'cuda_recovery.json':recovery,'platform_timing.json':platform,
              'checkpoint_diagnostics.json':diagnostics,'execution_review.json':review}
    for name in ['completion','dataset_validation','environment','execution_command','hardware','io_timing','preflight','depth_audit','storage_preflight','snapshot_index','wall_time','independent_recovery']:
        receipts[name+'.json']=artifacts/(name+'.json')
    for name,p in receipts.items():shutil.copyfile(p,evidence/name)
    (evidence/'receipt_checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.glob('*.json')},indent=2))
    cp=sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint')
    export=sum(r['seconds'] for r in io['records'] if r['operation']=='export')
    clipped=sum(r['gradient_norm']>1 for r in history)
    observations={'completed_updates':5000,'validation_count':20,'gradient_clipped_updates':clipped,
                  'gradient_clipping_fraction':clipped/5000,'learning_rates':sorted({r['lr'] for r in history}),
                  'checkpoint_seconds':cp,'snapshot_export_seconds':export,'journal_seconds':io['journal_seconds'],
                  'training_steps_per_second':5000/s['training_seconds'],
                  'training_tokens_per_second':s['training_tokens']/s['training_seconds'],
                  'pipeline_wall_seconds':t['pipeline_wall_seconds'],'archive_seconds':m['archive_seconds'],
                  'same_run_no_restart':d['failures']==d['resumes']==0,'generation_seconds':generations['seconds']}
    (out/'observations.json').write_text(json.dumps(observations,indent=2))
    text=f'''# E08 seed42 six-block attention experiment -- 2026-10-13

One explicitly authorized depth ablation: **5000 completed optimizer updates**,
seed42. Prior E01-E06 evidence is preserved; E07 Colab work is untouched.
No other seed, configuration or10k continuation was launched. The broader
three-seed study remains separately gated. Hypotheses and interpretations are
separate from observed measurements; superiority is not presumed.

## Research objective, hypothesis and capacity confound

Study the effect of changing four blocks to six at width192. **Hypothesis:**
additional sequential transformations and capacity might improve prediction,
but could increase cost or change optimization. Neither direction is assumed.
E08 has2694528 parameters, **889728 more than E01**. Depth, capacity and compute
change together; better scores cannot isolate depth independently of capacity.

## Architecture-specific correctness and initialization audit

Only n_layers changes4-to6. Each of six independent pre-LayerNorm blocks applies
x+dropout(attention(norm1(x))), then the MLP residual on norm2 of that result.
Explicit Q/K/V projections split to(batch,6,sequence,32), scale by sqrt(32),
apply causal masking/softmax/dropout and merge to192. GELU, sinusoidal encoding,
context128, MLP4 and dropout.1 are unchanged; no RoPE/learned positions.
Fresh seed42 initialization was used, never another experiment's weights.

CPU and actual CUDA audits passed13 checks: sole override/training controls,
six-block count, independent storage, exact parameter counts, seed42 repeat
construction, independent reference calculations through all six preLN/residual
paths, position/activation controls, six attention shapes/masking, finite loss
and manualCE, future-token invariance, finite/nonzero weight gradients in every
block, train/eval dropout and context-cropped generation. Per-block smoke
norms and observed shared E01 parameter equality are in depth_audit.json.
Matching seeds do not imply every initial parameter matches across depths:
shared first-four-block weights match, while the later output projection differs
because additional blocks consume initialization RNG. Different dropout/RNG
trajectories are expected. Numerical smoke gradients prove connectivity only,
not research vanishing/exploding gradients. No scientific algorithm was changed.

## Exact source, environment and reproducibility

- Reviewed main: `fad0afe2fc6259bd5fd2ddb5f004ce57e87bd34c`.
- Clean trained source: `{s['source_commit']}`.
- E01 trained source: `444bca4c4ac19f97f46fae210a9fd361f8561537`.
- Shared audited core hash: `{s['code_hash']}`.
- E08 resolved configuration hash: `{s['config_hash']}`.
- E08 unique research run: `{s['run_id']}`.
- [Private research notebook, saved version1](https://www.kaggle.com/code/{hw['kernel_id']}).
- [Private fresh recovery notebook](https://www.kaggle.com/code/{recovery_kernel}).
- GPU: {env['gpu']}, device0, {hw['gpu_count']} allocated devices;
  GPU0 total {hw['devices'][0]['total_bytes']:,} bytes.
- Python {env['python']}, PyTorch {env['torch']}, torch CUDA {env['cuda']}.
- OS: {env['platform']}. Full freeze and nvidia-smi/driver information: evidence.

The qualified Python3.11.14/torch2.5.1+cu124/NumPy1.26.4/transitive freeze was
reproduced; native Kaggle packages were untouched. E01/E08 freeze comparison passed
except editable Git SHA. Separate GPU sessions introduce clock/load/thermal and
instance variability; timing differences are descriptive, not isolated hardware
benchmarks. The source diff and tests reviewed orchestration only; no human PR
approval is represented as obtained. All code used for research was committed,
clean and CPU-CI tested before submission.

## Data, architecture, optimization and evaluation

Pinned Karpathy char-rnn revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e;
raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77;
vocabulary65. Source/splits/tokenizers/manifest match E01 exactly. Contiguous
train/validation/test bytes1003854/55770/55770; fit on training only and no windows
cross partitions. The test partition was not used for likelihood or selection.

Decoder-only pre-LayerNorm GPT: width192,6 blocks,6 heads,32 dimensions/head,
context128,MLP expansion4,GELU,dropout0.1,sinusoidal,**{s['parameters']:,} parameters**.
AdamW3e-4,constant schedule,weight decay0.01,gradient clip1,float32 CUDA,seed42,
microbatch4,accumulation8,4096 predicted targets/update,5000 optimizer updates.
No early stopping, AMP, LR change, architecture change or training-history restart.
Separate development checks precede a fully reseeded research run; they are not
counted as experimental optimizer updates or validation observations.

Full sliding validation every250 updates, complete55770-byte validation partition,
256 original conditioning bytes excluded,55514 target bytes/tokens scored once.
Conditional BPB=sum NLL/(ln2 * scored bytes), perplexity=exp(token NLL). Character
BPB equals token loss/ln2 here; these are related measures, not independent evidence.
No capped/sampled validation or invented step-zero measurement is included.
The existing checkpoint policy saves every validation state and every500-step
boundary, selects best on actual validation and exports immutable snapshots.

Train loss uses sampled batches/dropout; validation is eval-mode/full sliding.
Same-step gaps are noisy, unequal-protocol diagnostics. Training100-step trailing
means are labeled smoothing, not additional measurements. Equal tokens/steps do
not guarantee equal FLOPs or bytes of usable conditional context across variants.

## Genuine measured E08 results

| Quantity | Recorded value |
|---|---:|
| First training loss, update1 | {s['initial_recorded_training']['loss']:.8f} |
| First validation loss, update250 | {s['initial_recorded_validation']['loss']:.8f} |
| Final training loss | {s['final_training_loss']:.8f} |
| Final100-step trailing training mean | {s['final_training_trailing_100_loss']:.8f} |
| Best observed validation step | {s['best_validation']['step']} |
| Best validation loss / conditional BPB | {s['best_validation']['loss']:.8f} / {s['best_validation']['bpb']:.8f} |
| Final validation loss / conditional BPB | {s['final_validation']['loss']:.8f} / {s['final_validation']['bpb']:.8f} |
| Final validation perplexity | {s['final_validation']['perplexity']:.8f} |
| Processed predicted tokens / source bytes | {s['training_tokens']:,} / {s['training_source_bytes']:,} |
| Unique target source bytes | {s['unique_target_bytes']:,} |
| Gradient clipping updates | {clipped}/5000 ({clipped/50:.2f}%) |
| Peak allocated GPU memory,MiB | {s['peak_gpu_bytes']/2**20:.3f} |

Source-byte exposure includes repeats, not independent epochs. Uncovered target
source bytes: **{1003854-s['unique_target_bytes']:,}**, as reflected in the measured coverage;
the first source byte cannot be a target without a preceding token. Allocated
GPU memory excludes reserved/driver/device overhead.

| Update | Validation loss | Conditional BPB | Perplexity |
|---|---:|---:|---:|
'''
    for r in val:
        v=r['validation'];text+=f"| {r['step']} | {v['loss']:.8f} | {v['bpb']:.8f} | {v['perplexity']:.8f} |\n"
    text+='\n## E01 versus E08: matched observations\n\nDelta=E08 minus E01; negative loss/BPB differences favor E08. Relative differences are\npercentages of the E01 metric, not statistical effect-size estimates.\n\n| Metric | E01 | E08 | Delta | Relative,% |\n|---|---:|---:|---:|---:|\n'
    for r in compared['summary_comparison']:text+=f"| {r['metric']} | {r['E01']:.8f} | {r['E08']:.8f} | {r['delta_E08_minus_E01']:.8f} | {r['relative_percent']:.3f} |\n"
    text+='\n| Update | E01 loss | E08 loss | Delta loss | E01 BPB | E08 BPB |\n|---|---:|---:|---:|---:|---:|\n'
    for r in compared['matched_validation']:
        text+=f"| {r['step']} | {r['E01_loss']:.8f} | {r['E08_loss']:.8f} | {r['delta_loss']:.8f} | {r['E01_bpb']:.8f} | {r['E08_bpb']:.8f} |\n"
    text+='\nThe shared results collector validated both final checkpoint histories, metadata,\nconfiguration/data/tokenizer hashes and comparison policy. Final private sampling\nRNG states and source-byte coverage arrays also matched exactly. Only block count\ndiffers in configuration, adding889728 parameters and changing compute. One seed per configuration means sample SD is undefined,\nnot zero. Machine-readable comparison/CSV files preserve all20 matched validation\npoints and explicit incomplete-study seed counts. No E02 data are included.\n\n'
    text+='| E01 reference update | E01 recorded loss | First observed E08 update at/below it |\n|---|---:|---:|\n'
    for r in compared['retrospective_threshold_observations']:text+=f"| {r['E01_reference_step']} | {r['E01_reference_loss']:.8f} | {r['E08_first_observed_at_or_below_step']} |\n"
    text+='\nThese are retrospective thresholds on actual250-update observations, not exact\ncrossing times, interpolated measurements or a modified stopping rule. Both runs\ncompleted5000 updates irrespective of intermediate comparisons.\n\n'
    for name in ['loss','bpb','training','throughput','training_seconds','peak_gpu_bytes']:text+=f'![E01 vs E08 {name}](comparison/comparison_{name}.png)\n\n'
    text+='## Depth comparison framework\n\nE01 has4 blocks; E08 has6. E09 has8 blocks but was not authorized or trained here.\nIts metric fields remain null in [depth_framework.json](comparison/depth_framework.json).\nDepth and model capacity change together; no isolated depth-causality claim is made.\n\n'
    text+='## Separate positional-ablation context (not depth controls)\n\n| Experiment | Factor | Best conditional BPB | Best step |\n|---|---|---:|---:|\n'
    for eid, factor in [('E01','six-head sinusoidal baseline'),('E03','RoPE'),('E04','learned positions')]:
        context=read(root/f'report/pilots/{eid}_seed42/summary.json')
        text+=f"| {eid} | {factor} | {context['best_validation']['bpb']:.8f} | {context['best_validation']['step']} |\n"
    text+='\nE03/E04 change position, not depth; they are contextual observations, not\ncombined-factor controls. E04 adds24576 parameters and extra initialization RNG\ndraws. E08 versus E01 is the principal matched architecture comparison.\n\n'
    text+=f'''## Runtime, throughput and overhead

| Measured scope | Seconds |
|---|---:|
| Training only | {s['training_seconds']:.3f} |
| Full validation,20 evaluations | {s['validation_seconds']:.3f} |
| Checkpoint calls | {cp:.3f} |
| Verified snapshot export calls | {export:.3f} |
| Journal fsync/flush calls | {io['journal_seconds']:.3f} |
| Training CLI wall | {io['cli_wall_seconds']:.3f} |
| Archive compression and checksum scope | {m['archive_seconds']:.3f} |
| Notebook clone to archive-complete pipeline | {t['pipeline_wall_seconds']:.3f} |
| Complete genuine100-sample generation | {generations['seconds']:.3f} |

Training-only average: {observations['training_steps_per_second']:.4f} updates/sec,
{observations['training_tokens_per_second']:.2f} predicted tokens/sec. Generation
and archive scopes lie within the measured notebook pipeline; do not double-count
them. Pipeline timestamps exclude queue, final notebook conversion/output
publication and external download. Recovery evidence is recorded below; full network-transfer and
recovery wall time were not measured. Individual validation times were not logged; cumulative total is
genuine. No measured dollars or full-session lifetime is invented.

## E08 measured figures

Validation lines connect only20 actual observations. Training smoothing uses the
declared100-step trailing mean. Full PNG/PDF evidence is in the verified archive.

'''
    for name in ['training_loss','validation_loss','combined_loss','validation_bpb','generalization_gap','throughput','training_time','gradient_norm','learning_rate']:text+=f'![{name}](figures/{name}.png)\n\n'
    text+=f'''## Convergence descriptors and interpretation boundary

**Observation:** Actual3000 to 5000 validation loss improvement is
{s['loss_improvement_3000_to_5000']:.8f} nats/token. A descriptive least-squares
slope through actual3000-5000 points is {s['late_validation_slope_per_1000_steps']:.8f}
nats/token per1000 updates. Best observed step{s['best_validation']['step']}; final
is {s['final_validation']['loss']-s['best_validation']['loss']:.8f} above that minimum.
The subsequent reviewed interpretation discusses the shape, intervals, generation
and E01 comparison without treating small fluctuations as conclusive overfitting.
Observed best is selected from scheduled evaluations, not unknown intermediate
steps. No claim about double descent or long-context extrapolation is implied.

## Genuine fixed-protocol text generation

[Complete unedited100-sample grid](generation.md) and generation.json preserve all
five prompts, best/final checkpoints, greedy seed42 and temperatures0.7/1.0/1.2
with seeds42/123/456,256 new tokens. No external model generated or edited text.
Five predetermined best-checkpoint greedy outputs follow, without cherry-picking.
Qualitative comparisons use the same settings; coherent-looking output does not
establish statistical superiority, understanding or memorization absence.

'''
    for sample in generations['samples']:
        if sample['checkpoint_policy']=='best' and sample['greedy']:
            text+=f"### Prompt {sample['prompt']!r}\n\nCheckpoint `{sample['checkpoint']}`, greedy,seed42,256 new tokens.\n\n```text\n{sample['text']}\n```\n\n"
    text+=f'''## Publication, independent recovery and checkpoint manifest

Final checkpoint: `{c['final_checkpoint']}`.
Best checkpoint: `{c['best_checkpoint']}`.
All20 resumable weights and21 immutable snapshots are preserved. Exact optimizer,
constant scheduler,scaler,RNG and sampling/training history are included. No
restart, unrelated history concatenation or additional optimizer update occurred.
Fresh compatible Kaggle CUDA state roundtrip passed all audited state fields,
and best/final generation worked. Independent downloaded archive best/final CPU
inference also passed. Actual10k continuation was not started.

Archive bytes: **{m['archive_bytes']:,}**; SHA256 `{m['archive_sha256']}`.
All **{local['verified_file_count']} archive files** were independently CRC/path/
file-set/hash verified. [Complete manifest](evidence/artifact_manifest.json).
The archive stores complete research checkpoints once; full immutable snapshot
copies remain in published notebook output. Dataset/tokenizer manifest,resolved
configuration,environment,full5000-step history,20 validations,figures and genuine
generations are included. Small receipts are committed byte-for-byte; checkpoints,
datasets and full journals are excluded from GitHub.

| Checkpoint update | SHA256 |
|---|---|
'''
    for p,h in sorted(m['files'].items()):
        if '/checkpoints/' in p and p.endswith('.pt'):text+=f'| {int(Path(p).stem.split("_")[-1])} | `{h}` |\n'
    text+='''
## Scientific limitations and remaining scope

Single matched seed provides no significance test or across-seed SD. Validation
selects checkpoints; held-out test remains reserved. This is a fixed-context
depth comparison, not proof about other widths, depths or long-range
reasoning. Speed comparisons use separate sessions of the same GPU model, not
controlled repeated timing trials. Framework/seed matching does not guarantee
identical optimization trajectories when attention math changes. Greedy and
stochastic decoding quality are different diagnostics. Short phrase overlaps
may be ordinary Shakespeare language; they neither prove nor exclude memorization.
Baseline convergence cannot establish every architecture's training requirement.
Further seeds,configurations or10k extensions require separate authorization.
'''
    context_rows=[]
    for eid in ['E01','E03','E04']:
        context=read(root/f'report/pilots/{eid}_seed42/summary.json')
        context_rows.append({'experiment':eid,'parameters':context['parameters'],'best_bpb':context['best_validation']['bpb'],'best_step':context['best_validation']['step'],'single_seed':42})
    context_rows.append({'experiment':'E08','parameters':s['parameters'],'best_bpb':s['best_validation']['bpb'],'best_step':s['best_validation']['step'],'single_seed':42})
    (out/'contextual_bpb.json').write_text(json.dumps({'scope':'Separate single-factor seed42 pilots; no statistical ranking or combined-factor claim','rows':context_rows},indent=2)+'\n')
    text+='\n[Detailed evidence-backed interpretation and execution/retrieval instructions](interpretation.md)\ncomplete the qualitative analysis and recommendations.\n'
    (out/'E08_seed42_analysis.md').write_text(text,encoding='utf-8')
    print(out/'E08_seed42_analysis.md')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifacts','comparison','platform','recovery','manifest','diagnostics','review','root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-kernel',required=True)
    a=p.parse_args();report(a.artifacts,a.comparison,a.platform,a.recovery,a.manifest,a.diagnostics,a.review,a.root,a.output,a.recovery_kernel)
