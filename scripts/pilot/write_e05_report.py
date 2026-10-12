"""Render a detailed E05 report only from verified real artifacts."""
import argparse
import json
from pathlib import Path
import shutil


def report(root,evidence,comparison,output,links,qualitative):
    read=lambda p:json.loads(p.read_text())
    c=read(comparison/'comparison.json'); a=c['E01']; b=c['E05']; av=b['best_validation']; fv=b['final_validation']
    recovered=read(evidence.parent/'independent_verification.json')
    assert recovered['status']=='independent_download_reconstruction_all_hashes_and_CPU_inference_passed'
    pre=read(evidence/'preflight.json');hw=read(evidence/'hardware.json');env=read(evidence/'environment.json')
    audit=read(evidence/'relu_audit.json');grad=read(evidence/'gradient_diagnostics.json');act=read(evidence/'activation_diagnostics.json')
    archive=read(evidence.parent/'artifact_manifest.json');cuda=read(evidence/'cuda_recovery/recovery.json')
    checkpoints=read(evidence/'checkpoint_audit.json'); generation=read(evidence/'analysis/generation.json')
    files=read(links);notes=qualitative.read_text()
    timing=read(evidence/'pipeline_timing.json')
    output.mkdir(parents=True,exist_ok=False);(output/'figures').mkdir();(output/'evidence').mkdir()
    for p in (evidence/'analysis/figures').glob('*.png'):shutil.copyfile(p,output/'figures'/p.name)
    for p in (comparison/'figures').glob('*.png'):shutil.copyfile(p,output/'figures'/p.name)
    for name in ['summary.json','validation.csv','generation.json','generation.md']:shutil.copyfile(evidence/'analysis'/name,output/name)
    shutil.copyfile(comparison/'matched_validation.csv',output/'matched_validation.csv')
    for p in evidence.glob('*.json'):shutil.copyfile(p,output/'evidence'/p.name)
    shutil.copyfile(evidence/'cuda_recovery/recovery.json',output/'evidence/cuda_recovery.json')
    for name in ['artifact_manifest.json','archive_parts.json','independent_verification.json']:shutil.copyfile(evidence.parent/name,output/'evidence'/name)
    shutil.copyfile(comparison/'comparison.json',output/'evidence/comparison.json')
    shutil.copyfile(links,output/'evidence/artifact_links.json')
    text=f'''# E05 seed42 — ReLU activation Colab experiment

E05 completed exactly **5,000 optimizer updates**, **20 full sliding validations**, **20 complete validation checkpoint states**, and **{checkpoints['snapshot_count']} verified immutable research snapshots**. Best checkpoint: update **{av['step']}**, validation loss **{av['loss']:.8f}**, conditional BPB **{av['bpb']:.8f}**. Final update 5000: loss **{fv['loss']:.8f}**, BPB **{fv['bpb']:.8f}**. Durable publication, independent archive download/hash verification, genuine best/final inference and compatible CUDA optimizer/RNG restoration passed.

## 1. Objective, hypothesis and authorized scope

Question: how does replacing GELU with ReLU affect predictive performance, learning dynamics, computational efficiency and generated text in the original small GPT architecture? The hypothesis was that changing the feed-forward nonlinearity might alter learning and computational cost. No improvement or regression was presumed.

The user authorized E05 seed42 for exactly 5,000 updates. This is fresh initialization, with no E01/E02 weights loaded, no other research seed, factor combination, 10k extension or test likelihood. A separate two-update smoke run, restored and continued to smoke step 4, is excluded from scientific results. The full multi-seed study remains separately gated.

## 2. Technical difference and implementation audit

ReLU computes `max(0,x)`: negative inputs are gated to zero; its derivative is zero on negative inputs and one on positive inputs. PyTorch chooses zero gradient at exactly zero. GELU computes `x·Φ(x)`, weighting input by the standard normal CDF and preserving small negative outputs. These nonlinearities contain no trainable parameters. The canonical baseline uses exact default `nn.GELU()` and E05 uses default `nn.ReLU()`. Formula sources: [Hendrycks and Gimpel, GELUs](https://arxiv.org/abs/1606.08415), [PyTorch ReLU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [PyTorch GELU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html).

Potential computational savings and sparse activations were hypotheses. Validation and runtime measurements below determine the observed association. No mechanism is inferred from a validation difference alone.

The actual `TransformerBlock` MLP construction and strict `ModelConfig` resolver were inspected. Both CPU and qualified CUDA audits verified:

- Only resolved model difference from E01 is `activation: relu`; training defaults and character tokenizer unchanged.
- All four feed-forward blocks use actual `nn.ReLU`, and no `nn.GELU` exists in the E05 model.
- Each MLP remains 192→768→192; activation adds zero parameters. Total exactly 1,804,800.
- Shared LayerNorm, explicit Q/K/V causal attention, six heads, sinusoidal positions, embeddings, dropout and output projection retain their canonical code and shapes. No learned positions or RoPE.
- Every seed42 initial trainable parameter tensor equals independently constructed E01; all {audit['matching_state_tensors']} state tensors match. Initial parameter-byte SHA256 `{audit['initial_parameter_sha256']}`.
- For the known input `[-2,-1,0,1,2]`, all four actual ReLU modules produce `[0,0,0,1,2]`, with gradients `[0,0,0,1,1]`; outputs distinguish them numerically from GELU.
- Backpropagation through each entire actual MLP confirms zero gradient at sampled negative preactivation inputs, with finite and nonzero MLP weight gradients.
- Shifted next-token targets, manual cross-entropy equivalence, finite CUDA logits/loss/all parameter gradients, causal masks and invariance of earlier logits to altered future tokens passed.

[CUDA activation audit](evidence/relu_audit.json), [existing tests](evidence/tests.json), [activation-specific test](evidence/activation_test.json). Existing suite: 39 tests passed in 36.25s; added activation test passed in 0.65s. No implementation bug or scientific-source fix was required.

## 3. Exact source and environment

- Clean trained source: `{b['source_commit']}`, latest reviewed main when synchronized.
- Scientific source/config hash: `{b['code_hash']}`.
- Resolved E05 configuration hash: `{b['config_hash']}`.
- E01 reference trained source: `{a['source_commit']}`; canonical source/config diff is empty against this run. Reviewed changes introduced orchestration/reports only.
- E05 orchestration was separately versioned on its dedicated feature branch; the notebook records that commit. The installed `nano_lm` package points to the pinned clean main checkout. No notebook model/training loop exists.
- [Executed notebook]({files['notebook']}).
- Python `{env['python']}`, PyTorch `{env['torch']}`, torch CUDA `{env['cuda']}`; NumPy 1.26.4, tokenizers 0.21.0, pytest 8.3.4, Matplotlib 3.9.4. Exact 46-package E01/E02 transitive freeze matched, excluding editable project location.
- GPU `{env['gpu']}`, {hw['gpu_count']} visible device(s), training oncuda:0. Total GPU bytes {hw['devices'][0]['total_bytes']:,}. Full driver output and software freeze are preserved in [hardware](evidence/hardware.json) and [environment](evidence/environment.json).

Colab's native newer interpreter/framework was left intact and did not execute research training, evaluation or model generation. Isolated Python3.11.14/CUDA wheels were provisioned through the previously qualified workflow. `MPLBACKEND=Agg` was set before postprocessing.

E01 used Kaggle GPU0 of two allocated T4s, driver 580.178.04; this Colab run used driver 580.82.07. The same GPU model and dependency versions reduce some confounds, but host scheduling, driver, platform and storage differ. Runtime ratios are descriptive observations, not isolated activation speed effects.

## 4. Data, tokenizer and evaluation controls

Pinned Tiny Shakespeare revision `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`. Raw SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; character tokenizer SHA256 `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`. Vocabulary65. All source/split/tokenizer/manifest reference fields match E01. Original contiguous train/validation/test byte counts are1,003,854/55,770/55,770. Train-only tokenizer fitting and shifted windows preserve partition separation. [Verification](evidence/dataset_verification.json).

Validation used the entire 55,770-byte validation text, with the first 256 original bytes conditioning-only and exactly 55,514 target bytes/characters scored once per measurement. Full sliding evaluation batch 32/context 128 resets local positional indices exactly as E01. No cap, sampled validation or test likelihood. BPB is total NLL divided by `ln(2)×55,514`; with this identical ASCII character tokenizer it equals loss/ln(2). Both token cross-entropy and BPB compare directly across E01/E05. Perplexity is exp(loss). Best checkpoint selection minimizes full-validation loss.

## 5. Locked architecture and optimizer

Decoder-only from-scratch causal GPT: width 192, blocks 4, heads 6, head dimension 32, context 128, MLP expansion 4, **ReLU**, sinusoidal positions, dropout 0.1, vocabulary 65, **1,804,800 trainable parameters**. Explicit canonical attention is unchanged; no pretrained model or combined successful factors.

AdamW, constant LR 0.0003, weight decay 0.01, gradient clipping 1.0, seed42, float32, CUDA. Microbatch 4, accumulation 8, 4,096 predicted tokens/update; exactly 5,000 completed updates and zero AMP-skipped updates. Full validation every 250; nominal periodic checkpointing every 500 also saves every validation/best state. Optimizer/scheduler/scaler/RNG/sampling and complete histories are retained. [Execution settings](evidence/execution.json).

## 6. Actual learning curves and full matched validation

Initial recorded training loss: **{b['initial_recorded_training']['loss']:.8f}** at update 1. First full validation at 250: **{b['initial_recorded_validation']['loss']:.8f}**. Final sampled-batch loss: **{b['final_training_loss']:.8f}**; trailing 100 mean **{b['final_training_trailing_100_loss']:.8f}**. No step-zero validation was invented.

| Update | E01 loss | E05 loss | E01 BPB | E05 BPB | E05−E01 loss |
|---:|---:|---:|---:|---:|---:|
'''
    for r in c['rows']:text+=f"| {r['step']} | {r['E01_loss']:.8f} | {r['E05_loss']:.8f} | {r['E01_bpb']:.8f} | {r['E05_bpb']:.8f} | {r['loss_difference_E05_minus_E01']:+.8f} |\n"
    text+='''
[All 20 matched measurements](matched_validation.csv), [E05 loss/perplexity/BPB table](validation.csv).

![Training loss](figures/training_loss.png)
![Full validation loss](figures/validation_loss.png)
![Conditional BPB](figures/validation_bpb.png)
![E01 vs E05 validation loss](figures/e01_e05_validation_loss.png)
![E01 vs E05 BPB](figures/e01_e05_validation_bpb.png)

Lines connect observed validation endpoints; intermediate validation values are not measurements. The training smoother is descriptive. Sampled train batches use dropout, whereas validation uses evaluation mode; their gap alone is not evidence of overfitting. A single late fluctuation cannot establish deterioration or convergence.

## 7. E01-versus-E05 endpoint and efficiency comparison

| Metric | E01 GELU | E05 ReLU |
|---|---:|---:|
'''
    metrics=[('Parameters',1804800,1804800),('Best update',4750,av['step']),('Best validation loss',f"{a['best_validation']['loss']:.8f}",f"{av['loss']:.8f}"),('Best BPB',f"{a['best_validation']['bpb']:.8f}",f"{av['bpb']:.8f}"),('Final update',5000,5000),('Final validation loss',f"{a['final_validation']['loss']:.8f}",f"{fv['loss']:.8f}"),('Final BPB',f"{a['final_validation']['bpb']:.8f}",f"{fv['bpb']:.8f}"),('Final perplexity',f"{a['final_validation']['perplexity']:.8f}",f"{fv['perplexity']:.8f}"),('Training-only seconds',f"{a['training_seconds']:.3f}",f"{b['training_seconds']:.3f}"),('Full-validation seconds',f"{a['validation_seconds']:.3f}",f"{b['validation_seconds']:.3f}"),('Training tokens/sec',f"{20480000/a['training_seconds']:.3f}",f"{20480000/b['training_seconds']:.3f}"),('Optimizer updates/sec',f"{5000/a['training_seconds']:.6f}",f"{5000/b['training_seconds']:.6f}"),('Peak allocated GPU MiB',f"{a['peak_gpu_bytes']/2**20:.3f}",f"{b['peak_gpu_bytes']/2**20:.3f}"),('Clipping frequency',f"{100*c['E01_clipping_frequency']:.3f}%",f"{100*c['E05_clipping_frequency']:.3f}%"),('Target-token/source-byte exposure','20,480,000 / 20,480,000','20,480,000 / 20,480,000'),('Unique target bytes',a['unique_target_bytes'],b['unique_target_bytes'])]
    for label,x,y in metrics:text+=f'| {label} | {x} | {y} |\n'
    text+=f'''
Best loss/BPB relative reduction for E05: **{c['best_loss_reduction_percent']:+.4f}%**; final loss/BPB reduction **{c['final_loss_reduction_percent']:+.4f}%**. Positive reduction means E05 lower; negative means regression. Training-only time change **{c['training_time_change_percent']:+.3f}%**. Loss improvement from3k to5k: E01 {a['loss_improvement_3000_to_5000']:.8f}; E05 {b['loss_improvement_3000_to_5000']:.8f}. Descriptive late fitted loss slopes per 1,000 updates: E01 {a['late_validation_slope_per_1000_steps']:.8f}, E05 {b['late_validation_slope_per_1000_steps']:.8f}; no forecasting or significance inference.

This architectural comparison matches parameter count, tokenizer, source-byte context, initial parameter tensors, effective batch, optimizer and update budget. E02's parameter/context/exposure confounds do not apply here. Platform runtime confounds and single-seed uncertainty remain.

Actual original save/export calls were timed by forwarding their unchanged arguments and return values. Local checkpoint saves: **{c['checkpoint_seconds']:.3f}s**; verified Drive exports: **{c['Drive_export_seconds']:.3f}s**; journal writes/fsync: **{c['journal_seconds']:.3f}s**. Engine observed wall time through completed final export: **{c['engine_wall_seconds']:.3f}s**. These operations are separate from training/evaluation time. Archive construction/publication: **{archive['archive_and_publication_seconds']:.3f}s**. Observed notebook workflow envelope from the first recorded setup GPU timestamp to independent verification: **{timing['observed_notebook_workflow_wall_seconds']:.3f}s** (about {timing['observed_notebook_workflow_wall_seconds']/60:.2f}minutes). This includes provisioning, mount, preflight, research, interactive pauses, recovery, generation, publication and retrieval, but excludes later report/PR work. It is derived from recorded clock endpoints with one-second start precision and different-host clock-source caveats, rather than a single continuous timer. Finish phase: **{timing['finish_phase_seconds']:.3f}s**; independent integrity/inference verification excludes download and took **{timing['independent_verification_seconds_excluding_download']:.3f}s**. [Timing provenance](evidence/pipeline_timing.json). Peak allocated memory excludes driver overhead/reserved memory. No FLOP measurement exists.

![Throughput](figures/throughput.png)
![Accumulated training time](figures/training_time.png)

## 8. Gradient and measured activation behavior

All 5,000 recorded gradient norms are finite. Before clipping: mean{grad['norm_mean']:.6f}, median{grad['norm_median']:.6f}, range{grad['norm_min']:.6f}–{grad['norm_max']:.6f}, final{grad['final_norm']:.6f}. {grad['steps_above_clipping_threshold']} updates exceeded1.0 ({100*grad['clipping_frequency']:.3f}%), compared with E01's 0.140%. E01's complete norm aggregate is not in the committed summary; its original figure and recorded clipping fraction are the reference. [E05 diagnostics](evidence/gradient_diagnostics.json).

![Gradient norms](figures/gradient_norm.png)
![Learning rate](figures/learning_rate.png)

After training, best/final ReLU outputs were measured without optimizer updates on four predetermined 128-character training windows starting 0,10000,100000,900000 (512 positions), in evaluation mode. These are real activation statistics on a finite diagnostic sample, not validation/test selection.

| Checkpoint policy | Block | Zero-output fraction | Positive fraction | Units inactive throughout this sample |
|---|---:|---:|---:|---:|
'''
    for obs in act['observations']:
        for r in obs['blocks']:text+=f"| {obs['policy']} | {r['block']+1} | {r['zero_output_fraction']:.6f} | {r['positive_fraction']:.6f} | {r['sample_inactive_unit_count']}  / 768 |\n"
    text+='''
[Full activation evidence](evidence/activation_diagnostics.json). Zero activations are expected for ReLU. A unit inactive on512 positions is not proven permanently dead; these data do not establish a dying-ReLU problem or its causal contribution. There is no longitudinal all-corpus unit-activity audit.

## 9. Genuine best/final text grid and qualitative observations

The unchanged `configs/generation_protocol.json` defines five prompts, greedy decoding, temperatures 0.7/1.0/1.2, seeds 42/123/456 and 256 new character tokens. Both checkpoint policies supply 50 settings each: **100 unedited labeled records**. [Complete JSON grid](generation.json), [complete readable grid](generation.md). Every record names the real checkpoint and checksum. No other language model generated or rewrote these outputs.
'''
    if av['step']==5000:text+='\nBest and final are the same step5000 weights; their matching outputs are duplicate labels and supply 50 unique settings, not100 independent samples.\n'
    else:text+=f"\nBest step{av['step']} and final step5000 are distinct weights; both complete grids are retained.\n"
    text+='\n'+notes.split('### Activation observations and interpretation')[0]+'\n\n### Predetermined best-checkpoint greedy examples\n'
    for s in generation['samples']:
        if s['checkpoint_policy']=='best' and s['greedy']:
            text+=f"\n#### Prompt {s['prompt']!r}\n\n```text\n{s['text']}\n```\n"
    text+='\n### Full-grid descriptive repetition comparison\n\n| Experiment | Policy | Decoding | Samples | Mean repeated8-gram position fraction |\n|---|---|---|---:|---:|\n'
    for r in c['generation_repetition']:text+=f"| {r['experiment']} | {r['policy']} | {r['mode']} | {r['samples']} | {r['mean_repeated_8gram_position_fraction']:.6f} |\n"
    text+=f'''
Repetition counts use character8-gram positions in the256-character continuation, excluding prompt. They are descriptive indicators, not fluency or narrative quality scores. Temperature trades repetition against diversity/error and cannot substitute for loss-based evaluation. Entire grids remain available to inspect unfavorable samples.

## 10. Persistent storage, archive and recovery

Before research, actual Drive quota and a fresh exclusive E05 directory were verified. Smoke checkpoint size{pre['smoke_checkpoint_bytes']:,}bytes; conservative storage budget{pre['conservative_artifact_budget_bytes']:,}bytes; actual quota free approximately{pre['actual_drive_quota_free_bytes']:,}bytes as shown in Drive UI. Drive FUSE disk usage was not treated as account quota. All 3 smoke manifests/file hashes verified; fresh-directory checkpoint checksum, actual CUDA loading/inference and complete-state continuation passed.

Research run: `{b['run_id']}`. All 20 validation checkpoint states and extra completed-summary snapshot survived verified export. Every checkpoint's optimizer counters, scheduler epoch, history length and byte/token exposure were audited. Research failure/resume events: **{len(checkpoints['journal_failures_or_resumes'])}**; histories retain actual events. [Checkpoint audit](evidence/checkpoint_audit.json), [snapshot index](evidence/snapshot_index.json).

Fresh compatible CUDA process restored model, AdamW, scheduler, scaler, Python/NumPy/torch/CUDA/sampling RNG and training state. A save/read roundtrip compared all state structures exactly; all passed. Best/final GPU inference ran from restored snapshots, with **zero additional optimizer updates**. [CUDA recovery receipt](evidence/cuda_recovery.json).

- [Private durable E05 folder]({files['folder']}).
- [Complete archive]({files['archive']}): **{archive['archive_bytes']:,}bytes**.
- Archive SHA256: `{archive['archive_sha256']}`.
- [Archive manifest](evidence/artifact_manifest.json), [ordered part checksums](evidence/archive_parts.json).

The archive was published using the tested staged-rename publisher, then read back and hash-verified. Every internal entry's CRC/hash was checked. Numbered parts under the connector's transfer cap were independently downloaded and concatenated. Independent local checks verified exact archive SHA256, {recovered['archive_entries']}entry hashes/paths/CRC, all20 actual checkpoint optimizer/scheduler states,5000 histories and20 full validations. Authentic generation checkpoint hashes and best/final CPU inference passed. [Independent retrieval receipt](evidence/independent_verification.json).

Local CPU inference used Python{recovered['CPU_environment']['python']} / torch{recovered['CPU_environment']['torch']} for inference migration only. Exact training resume was validated on the locked CUDA environment, not this newer CPU framework. Large weights and datasets remain outside GitHub. The pinned manifest permits data reacquisition without including source text in the archive.

## 11. Failures and scientific limitations

Initial notebook authoring produced an unterminated string syntax error before any setup or training executed. The cell was corrected and the environment provisioned successfully. Host pytest was unavailable; qualified Colab executed the real test suite and new activation test. These preflight/authoring observations are separated from research logs. Independent local part00 download encountered HTTP/2 INTERNAL_ERROR92 after a partial transfer; part01 completed and matched its hash. Part00 resumed over HTTP/1.1 through a refreshed authenticated file reference, then both hashes and the complete reconstructed archive passed. The incomplete transfer was not counted as preservation evidence. [Failure history](evidence/failures.json). No unresolved execution or preservation failure remains.

One seed supplies no variance estimate or statistical significance. Only the original architecture/budget was tested. Equal initial weights and sample budget improve control, but distinct nonlinearities lead to distinct gradient trajectories. Kaggle/Colab driver/host differences limit runtime attribution. Evaluation is conditional sliding likelihood, not whole-file entropy. Generated text coherence is qualitative;8-gram repetition is not a human quality measure. Limited post-training activation samples cannot establish permanent inactive units or causal failure mechanisms. No test likelihood, unseen-factor interaction or5k→10k continuation was performed.

## 12. Interpretation and implications

{notes.split('### Activation observations and interpretation')[-1].replace('## Conclusion','### Conclusion')}

E02 best BPB2.02659521, E03 RoPE2.07327076 and E04 learned positions2.23778869 provide context for separately varied factors. They were not combined into E05. This run does not authorize a combined optimized configuration. Preserve this endpoint and review the evidence before additional seeds/extensions.

## 13. Collaboration and delivery

Dedicated E05 branch/report/notebook/Drive namespace. Primary researcher owns E06; a coordination comment on its PR records the separation. E01–E04 reports/artifacts and E06 branch were untouched; canonical scientific files remain unchanged. [Review PR]({files['pr']}) includes orchestration, correctness test, small evidence, tables, plots and documentation. No merge was performed.
'''
    (output/'E05_seed42_analysis.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','evidence','comparison','output','links','qualitative']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.root,a.evidence,a.comparison,a.output,a.links,a.qualitative)
