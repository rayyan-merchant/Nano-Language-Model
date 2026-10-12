"""Render the E02 single-seed report from retrieved, verified measurements."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def write_report(evidence, reference, output):
    read=lambda name:json.loads((evidence/name).read_text())
    s=read('summary.json'); a=read('checkpoint_audit.json'); context=read('source_byte_context.json')
    gen=read('generation.json'); archive=read('artifact_manifest.json'); independent=read('independent_verification.json')
    e=json.loads((reference/'summary.json').read_text())
    rows=list(csv.DictReader((evidence/'validation.csv').open()))
    assert s['completed_optimizer_updates']==5000 and len(rows)==20 and len(gen['samples'])==100
    assert independent['all_archive_file_hashes_match']
    output.mkdir(parents=True,exist_ok=True); (output/'evidence').mkdir(exist_ok=True)
    for name in ['summary.json','validation.csv','generation.json','generation.md']:
        shutil.copyfile(evidence/name,output/name)
    for name in ['preflight.json','research_launch.json','checkpoint_audit.json','source_byte_context.json',
                 'snapshot_index.json','recovery_environment.json','training_environment.json','artifact_manifest.json',
                 'archive_parts.json','independent_verification.json','failures.json','code_equivalence.json',
                 'environment_hashes.json','preflight_test_receipt.json']:
        shutil.copyfile(evidence/name,output/'evidence'/name)
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    for p in (evidence/'figures').glob('*.png'): shutil.copyfile(p,figures/p.name)
    erows=list(csv.DictReader((reference/'validation.csv').open()))
    fig,ax=plt.subplots(figsize=(8,4.5))
    for records,label in [(erows,'E01 character, seed 42'),(rows,'E02 Byte-BPE, seed 42')]:
        ax.plot([int(r['step']) for r in records],[float(r['bpb']) for r in records],marker='o',markersize=3,label=label)
    ax.set(xlabel='Completed optimizer updates',ylabel='Conditional validation bits per original source byte',
           title='Same 55,514 scored bytes; 256-byte conditioning prefix')
    ax.grid(alpha=.2);ax.legend()
    fig.text(.5,.01,'Single seed; parameter counts, source-byte exposure and byte context differ.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(figures/'e01_e02_validation_bpb.png',dpi=200);plt.close(fig)
    selected=[x for x in gen['samples'] if x['checkpoint_policy']=='best' and x['greedy']]
    diagnostic=[]
    for label in ['greedy',.7,1.,1.2]:
        samples=[x for x in gen['samples'] if x['checkpoint_policy']=='best' and
                 (x['greedy'] if label=='greedy' else not x['greedy'] and x['temperature']==label)]
        continuations=[x['text'][len(x['prompt']):] for x in samples]
        repeated=[]
        for text in continuations:
            grams=[text[i:i+8] for i in range(max(0,len(text)-7))]
            repeated.append(1-len(set(grams))/len(grams) if grams else 0)
        diagnostic.append({'setting':str(label),'samples':len(samples),'mean_continuation_characters':statistics.mean(map(len,continuations)),
                           'mean_repeated_8gram_position_fraction':statistics.mean(repeated),'replacement_characters':sum(t.count('\ufffd') for t in continuations)})
    (output/'generation_diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    best=s['best_validation']; final=s['final_validation']; delta=e['best_validation']['bpb']-best['bpb']
    comparisons=[
        ('Model parameters',f"{e['parameters']:,}",f"{s['parameters']:,}"),
        ('Vocabulary',65,1024),('Optimizer updates',5000,5000),
        ('Predicted training tokens',f"{e['training_tokens']:,}",f"{s['training_tokens']:,}"),
        ('Training source-byte exposure, repeats included',f"{e['training_source_bytes']:,}",f"{s['training_source_bytes']:,}"),
        ('Unique target bytes',f"{e['unique_target_bytes']:,}",f"{s['unique_target_bytes']:,}"),
        ('128-token training context, original bytes','128',f"mean {context['training_128_token_context_bytes']['mean']:.3f}; range 211–401"),
        ('Validation preceding context, original bytes','128',f"mean {context['validation_scored_target_preceding_context_bytes']['mean']:.3f}; range 204–405"),
        ('Best validation BPB',f"{e['best_validation']['bpb']:.8f}",f"{best['bpb']:.8f}"),
        ('Final validation BPB',f"{e['final_validation']['bpb']:.8f}",f"{final['bpb']:.8f}"),
        ('Best-checkpoint update',e['best_validation']['step'],best['step']),
        ('Training-only seconds',f"{e['training_seconds']:.3f}",f"{s['training_seconds']:.3f}"),
        ('Full validation seconds, 20 evaluations',f"{e['validation_seconds']:.3f}",f"{s['validation_seconds']:.3f}"),
        ('GPU','Tesla T4; GPU0 used of two allocated','Tesla T4; one visible in preflight'),
        ('Generation behavior','Speaker turns; greedy loops and malformed words','Speaker turns and readable phrases; greedy loops, malformed subword blends; one unique replacement-character sample'),
        ('Limitations','One seed; 128-byte context; character exposure','One seed; 20.40% more parameters; ~2.44× byte exposure and longer byte context'),
    ]
    table='| Measure | E01 seed42 | E02 seed42 |\n|---|---:|---:|\n'+'\n'.join(f'| {k} | {x} | {y} |' for k,x,y in comparisons)
    validation_table='| Update | Token validation NLL | Perplexity | Conditional BPB |\n|---:|---:|---:|---:|\n'+'\n'.join(f"| {r['step']} | {float(r['loss']):.8f} | {float(r['perplexity']):.8f} | {float(r['bpb']):.8f} |" for r in rows)
    samples='\n'.join(f"\n### Fixed greedy prompt {i+1}: {x['prompt']!r}\n\n```text\n{x['text']}\n```\n" for i,x in enumerate(selected))
    diag_table='| Best-checkpoint setting | Samples | Mean continuation characters | Mean repeated 8-gram position fraction | Replacement characters |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {r['setting']} | {r['samples']} | {r['mean_continuation_characters']:.1f} | {r['mean_repeated_8gram_position_fraction']:.4f} | {r['replacement_characters']} |" for r in diagnostic)
    report=f'''# E02 seed42 — Byte-BPE Colab pilot

E02 completed **5,000 optimizer updates**, **20 full sliding validations**, **20 resumable validation checkpoint states**, and **21 verified immutable snapshots**. Fresh-runtime recovery, actual best/final GPU inference, the complete fixed generation grid, and independent archive retrieval passed. The best and final checkpoint are both update **{best['step']}**, with conditional validation BPB **{best['bpb']:.8f}**. PR publication is a separate review action; this report does not imply a merge or wider study approval.

## 1. Objective, hypothesis, and scope

This single authorized pilot tests the locked E02 tokenizer change against the [committed E01 seed42 reference](../E01_seed42/E01_seed42_analysis.md). The hypothesis was that train-only Byte-BPE may model reusable subword units more effectively than characters under the shared training protocol. The observed same-byte conditional validation score is lower for this E02 run. This is one observation with one seed and substantial parameter, context, exposure, and compute confounds; it does not establish a definitive tokenizer winner or statistical significance.

Authorization covered E02 seed42 for exactly 5,000 completed updates. No additional research configuration, seed, or 10k continuation was launched. The separate E02 preflight smoke restored at step 2 and continued to smoke step 4; it is excluded from every research table. The study-wide NO-GO remains.

## 2. Source and software equivalence

- Pinned clean training commit: `{s['source_commit']}` (main after PR #4 was already merged).
- E01 trained commit: `{e['source_commit']}`.
- Both canonical source/configuration hashes: `{s['code_hash']}`.
- `git diff` across `src`, `configs`, and `experiments/configs` was empty. Changes after the E01 trained commit were reviewed orchestration/report changes merged through PR #4.
- E02 resolved configuration hash: `{s['config_hash']}`.
- Canonical training environment SHA256: `{read('environment_hashes.json')['training_environment_canonical_sha256']}`. The normalized dependency-freeze hashes also match E01; see [environment hashes](evidence/environment_hashes.json).
- Python 3.11.14, PyTorch 2.5.1+cu124, CUDA 12.4, NumPy 1.26.4, tokenizers 0.21.0, pytest 8.3.4, matplotlib 3.9.4, and the exact 46-package E01 transitive freeze were qualified in an isolated environment.
- Colab's native Python 3.13.15 / PyTorch 2.11.0+cu130 remained installed. It did not execute research training. Standard-library archive finalization used the notebook's native Python; likelihood, recovery model inference, generation and figures used the qualified isolated environment.
- T4 CUDA matrix forward/backward and finite-gradient checks passed. The full automated suite passed: **35 tests in 30.96 seconds**.

The E02 checkpoint was freshly initialized with seed 42. No E01 weights were loaded. The committed analysis scripts import the shared package. The [thin launcher](../../../notebooks/colab_e02_seed42_pilot.ipynb) records the source and commands, with research execution disabled by default. [Executed Colab notebook](https://colab.research.google.com/drive/1Jiu1tKIHBN714tDjB4nu_6YGGE0-DSCJ).

## 3. Dataset and tokenizer provenance

Pinned Tiny Shakespeare revision: `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`.

| Artifact | Verified value |
|---|---|
| Raw source SHA256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Dataset manifest hash | `76b594a1979107795734f8d2cda08b56c1520c8d44883dbaa5a61dbee53ba16f` |
| BPE tokenizer SHA256 | `2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55` |
| Validation source SHA256 | `c64e90c160750d91a9d2d9b1ace5eab1a440f5a1701d32a8e97a179b7839f1c8` |
| Original bytes: train / validation / test | 1,003,854 / 55,770 / 55,770 |
| BPE tokens: train / validation / test | 411,158 / 24,516 / 24,905 |
| Vocabulary | 1,024 |

Preparation and independent validation used the repository implementation. Contiguous partitions precede fitting and tokenization. BPE was fitted on the training text only; all partitions round-tripped losslessly with byte conservation. Held-out preprocessing checks establish encoding integrity, not test-set model selection. Training windows use only training IDs, with shifted next-token targets and no partition crossing. No test likelihood was evaluated.

## 4. Architecture and training/evaluation protocol

The from-scratch decoder-only causal Transformer has width 192, 4 blocks, 6 attention heads, a 128-token context, a 4× MLP, GELU, sinusoidal positions, and dropout 0.1. Explicit shared attention/model code is unchanged. E02 has **2,173,056 parameters**, versus E01's 1,804,800; vocabulary embedding/output growth adds 368,256 parameters (**20.40%**).

AdamW used constant LR 3e-4, weight decay 0.01, norm clipping 1.0, seed 42, float32, CUDA, microbatch 4, accumulation 8, and 4,096 predicted tokens per update. The run stopped at exactly 5,000 completed optimizer updates with zero skipped updates. Validation every 250 and nominal checkpoint cadence every 500 also preserved every validation/best state.

Conditional validation BPB follows [docs/bpb.md](../../../docs/bpb.md): the identical 55,770-byte validation text, first 256 original bytes as conditioning, exactly 55,514 scored bytes, separately tokenized prefix/suffix to force the common byte boundary, full sliding context, each suffix target scored once, reset local positions, and no future-token leakage. E02 scores 24,386 suffix tokens. NLL is divided by `ln(2) × 55,514`, rather than token count, for the cross-tokenizer metric. Prefix-boundary segmentation is constrained and is not unconstrained corpus entropy.

Token NLL and perplexity below are within-E02 diagnostics. Their units differ from E01 and must not be ranked across tokenizers. Best selection minimized the recorded full-validation token NLL; with this fixed E02 segmentation and denominator, it also minimized BPB.

## 5. Measured learning curves and checkpoint metrics

Initial recorded step loss: {s['initial_recorded_training']['loss']:.8f}. Final batch loss: {s['final_training_loss']:.8f}; final trailing-100 training mean: {s['final_training_trailing_100_loss']:.8f}. Best/final validation NLL: {final['loss']:.8f}; perplexity: {final['perplexity']:.8f}; BPB: {final['bpb']:.8f}.

{validation_table}

![E02 training loss](figures/training_loss.png)
![E02 validation loss](figures/validation_loss.png)
![E02 combined token losses](figures/combined_loss.png)
![E02 conditional BPB](figures/validation_bpb.png)
![Same-byte E01/E02 BPB comparison](figures/e01_e02_validation_bpb.png)

Lines join recorded validation points; no intermediate validation measurements were invented. Validation improved from BPB 2.09669180 at 3k to 2.02659521 at 5k, a decrease of 0.07009659. Small increases occurred from 3,750 to 4,000 and from 4,500 to 4,750. The final point was best; the run does not demonstrate a plateau. The within-E02 3k-to-5k token-NLL decrease was {s['loss_improvement_3000_to_5000']:.8f}. A descriptive fitted late NLL slope was {s['late_validation_slope_per_1000_steps']:.8f} per 1,000 updates; it is not a forecast or significance test.

The final validation minus trailing-100 training NLL is {final['loss']-s['final_training_trailing_100_loss']:.8f}. Dropout, sampled batches, and evaluation mode differ, so this gap alone does not prove overfitting. Further approved continuation could clarify convergence, but none was started.

## 6. Throughput, exposure, and resource use

| Recorded quantity | E02 value |
|---|---:|
| Training-only seconds | {s['training_seconds']:.3f} |
| Full-validation seconds, 20 measurements | {s['validation_seconds']:.3f} |
| Engine segment wall time through summary creation | 1,371.960 |
| Remaining setup/journal/checkpoint/export overhead, unattributed | 44.490 s |
| Predicted tokens / training second | {s['training_tokens']/s['training_seconds']:.3f} |
| Optimizer updates / training second | {5000/s['training_seconds']:.6f} |
| Source-byte exposure / training second | {s['training_source_bytes']/s['training_seconds']:.3f} |
| Peak PyTorch allocated GPU memory | {s['peak_gpu_bytes']/2**20:.3f} MiB |
| Training predicted-token exposure | {s['training_tokens']:,} |
| Training source-byte exposure, repeats included | {s['training_source_bytes']:,} |
| Unique target bytes | {s['unique_target_bytes']:,} / 1,003,854 |
| Exposure bytes / predicted token | {context['training_exposure_bytes_per_predicted_token']:.8f} |
| Fresh snapshot verification/restore/counter audit | {a['snapshot_verification_restore_audit_seconds']:.3f} s |
| Complete fixed-grid GPU generation | {gen['seconds']:.3f} s |

The 44.490-second residual includes setup and multiple IO operations and is not an individually measured save/export timing. Final post-summary export, environment installation, archive construction/retrieval and later recovery are outside the engine segment timing. Per-checkpoint/export timings and FLOPs were not recorded and remain unavailable. PyTorch allocated-memory peaks exclude driver and process overhead.

All possible sampled training 128-token contexts span mean **312.514 original bytes**, median 313, range 211–401. Canonical validation contexts span mean **291.128 bytes**, range 204–405. E01 spans 128 ASCII bytes. E02 therefore has longer effective byte context and **2.441×** repeated target-byte exposure at equal update/token counts. Unique target coverage was {context['training_unique_byte_coverage_fraction']*100:.6f}%; these are sampled-target coverage counts, not FLOPs, unique epochs, or complete training-input traffic.

Both runs used the Tesla T4 model and matched software pins. E01 allocated two GPUs but trained on GPU0; E02's preflight exposed one. Kaggle versus Colab, host/driver and runtime conditions remain uncontrolled. E02 training-only time was 23.15% higher, while full-validation time was lower. Fewer BPE validation targets are a relevant workload difference. These timings do not isolate a causal tokenizer speed effect.

![Training throughput](figures/throughput.png)
![Gradient norm](figures/gradient_norm.png)
![Learning rate](figures/learning_rate.png)

## 7. Genuine fixed-protocol generation

The shared generation protocol used five fixed prompts, greedy decoding, temperatures 0.7/1.0/1.2, seeds 42/123/456, and 256 new tokens. Best and final produced 100 labeled records (50 settings for each checkpoint policy). Both policies refer to the same step-5,000 weights here and their outputs match. The entire unedited grid is preserved in [generation.json](generation.json) and [generation.md](generation.md). No other language model generated or repaired these outputs.

The five predetermined greedy samples follow in full. They show speaker labels and locally readable phrases, followed by substantial repetition. The Romeo and citizen samples loop around “I'll tell you”; the king sample repeats field/kingdom language and invents word forms.

{samples}

Temperature 0.7 sometimes yields valid local dialogue phrases but also forms such as `sadsop`, `discocourse`, and `mippositions`. Higher temperatures diversify continuations and introduce further malformed word/subword blends. One unique setting—`To be, or not to be`, temperature 1.2, seed 456—contains a replacement character in `MAPSURE�ber`. It appears once in each identical best/final label. Dataset round-trip integrity passed; an unconstrained generated byte sequence can still yield invalid UTF-8 on decoding. This artifact was preserved unchanged.

{diag_table}

These descriptive repeated-character-window counts are within-E02 diagnostics and are not a quality score. Lower repetition at higher temperature does not prove improved coherence. Equal 256-token generation budgets produce longer BPE text than character text: mean greedy continuation length is 632.2 characters for E02. The E01 predetermined samples also loop and contain malformed words. E02 sometimes displays longer readable phrases, but stable word, narrative, or semantic coherence is not consistently established under these settings. More fluent-looking passages are not evidence of superior generalization by themselves.

## 8. Failures, durable storage, and recovery evidence

Drive access was initially declined, unmounted, then explicitly approved by the user. The exploratory direct Colab download did not reach the computer; the approved Drive route was used. Before research, a unique directory was created, writes checked, Drive Storage showed 8.03 GB of 400 GB used, three smoke snapshot manifests/hashes verified, and a fresh-directory checkpoint loaded, generated, and resumed on CUDA. A conservative 100-checkpoint storage budget was 2,722,343,600 bytes.

The training engine published verified immutable snapshots at every validation boundary, so the maximum exposure since a completed durable export was 250 updates plus an export in progress. All 20 validation states and the extra final-summary snapshot are preserved. The original local runtime disappeared after the completed log and snapshots were saved; the cause was not measured. The fresh matching runtime restored all checkpoint files solely from exact snapshot hash matches. It verified all 20 optimizer counters, scheduler epochs, history lengths, complete 5k journal, evaluation policy and byte counts. The research journal recorded no failure or resume events. Actual best/final GPU inference was executed after independent restores, with zero additional optimizer updates.

Postprocessing initially failed at final manifest publication because `atomic_json` used a hard link unsupported by mounted Drive. Generation and the complete ZIP had already succeeded. A staged-rename manifest publisher repaired this without repeating training or generation. The failure log remains stored. One subsequent repair-script upload did not arrive and its file-existence guard failed; direct standard-library archive verification then succeeded. These are postprocessing/transfer failures, not missing model updates.

The connector rejected the 477,310,968-byte ZIP because its per-download limit is 268,435,456 bytes. The same archive was split into checksum-tracked parts, independently downloaded, concatenated, and verified. ZIP paths, duplicate entries, CRC, all {len(archive['files'])} archive file hashes, all 20 checkpoint files, and final checkpoint hash passed independent local checks. See [independent verification](evidence/independent_verification.json), [checkpoint audit](evidence/checkpoint_audit.json), [snapshot index](evidence/snapshot_index.json), and [failures](evidence/failures.json).

Independent local CPU inference used Python {independent['CPU_inference']['python']} / PyTorch {independent['CPU_inference']['torch']} for inference migration only, with the shared package and tokenizer 0.21.0. From the recovered checkpoint it actually produced `{independent['CPU_inference']['text']!r}` for the fixed recovery prompt `To be` and 16 greedy new tokens. No local training continuation occurred. Local archive verification/inference took {independent['verification_seconds']:.3f} seconds, excluding download. Transfer and archive-construction timings were not measured separately.

## 9. E01 versus E02 scientific comparison

{table}

Best E02 BPB is lower by **{delta:.8f} bits/byte ({100*delta/e['best_validation']['bpb']:.4f}%)** than the committed E01 best. Both score the identical validation hash and same 55,514 original-byte suffix with the identical conditioning/evaluator policy and core source. Direct BPB comparison is therefore appropriate within this conditional policy. The advantage is associated with this full E02 configuration; more parameters, longer byte context and ~2.44× source-byte exposure prevent attributing it solely to BPE merges. Equal updates do not equal raw-text exposure, FLOPs, cost or effective context. One seed per configuration supplies no variance estimate, significance claim or definitive winner.

## 10. Storage and retrieval

Run ID: `{s['run_id']}`.

- [Private durable artifact directory](https://drive.google.com/drive/folders/16g4kvL_hYt6GDG-UIgVBDkD0-IW_t8-T).
- [Immutable research snapshots](https://drive.google.com/drive/folders/1vy59z7Mz1xyLHiEs93fqhfSar2Uj9euw).
- [Complete archive](https://drive.google.com/file/d/1js-oI8f4zew4R_Ix48KYov21YGbo22TP/view), {archive['archive_bytes']:,} bytes.
- Archive SHA256: `{archive['archive_sha256']}`.
- Best/final checkpoint SHA256: `{independent['final_checkpoint_sha256']}`.
- [Complete archive manifest](evidence/artifact_manifest.json) and [ordered part checksums](evidence/archive_parts.json).

Download the full ZIP through an authenticated client supporting its size, or download the numbered parts in `archive_parts` and concatenate them in numeric order. Verify the complete SHA256, ZIP CRC and each manifest entry before loading. The archive contains the complete recovered run, all 20 checkpoints, histories, tokenizer, dataset manifest, optimizer/scheduler/RNG/sampling state, figures, generations and recovery evidence. The pinned source plus dataset manifest allow original-data reacquisition; large weights and dataset files remain outside GitHub.

To restore a snapshot, use the shared `nano_lm.storage verify` and `restore` commands into a fresh directory. Future training continuation requires matching PyTorch/GPU, code, dataset, tokenizer and resolved training settings. The final saved state is sufficient for a separately approved 10k total target; no continuation is authorized by this report. CPU inference migration does not imply exact training-resume equivalence on another framework or device.

## 11. Recommendations and completion status

Preserve this 5k endpoint and evaluate additional approved seeds only after independent environment/storage gates are reconciled. Keep the locked optimization and evaluation protocol for the planned ablations. Treat final-is-best and the remaining late BPB improvement as reasons to review the 5k budget, not to extend it automatically. Any future matched-byte/context or parameter-controlled experiment would be a new design requiring approval. Sample looping and malformed words remain substantive model limitations even with lower BPB.

The scientific execution, required validations, all checkpoint states, durable exports, independent recovery/inference, generation grid, plots and analysis are complete. Publication requires opening the review PR; no branch merge has been performed. Every value here derives from recorded evidence; per-checkpoint IO timings and FLOPs are explicitly unavailable.
'''
    (output/'E02_seed42_analysis.md').write_text(report,encoding='utf-8')
    print(output/'E02_seed42_analysis.md')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['evidence','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();write_report(a.evidence,a.reference,a.output)
