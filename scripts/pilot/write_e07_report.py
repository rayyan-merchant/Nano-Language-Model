"""E07 research report from independently verified genuine metrics only."""
import argparse
import json
from pathlib import Path
import shutil


def report(root,evidence,comparison,output,links,qualitative):
    read=lambda p:json.loads(p.read_text())
    c=read(comparison/'comparison.json');a=c['E01'];b=c['E07'];best=b['best_validation'];final=b['final_validation']
    verified=read(evidence.parent/'independent_verification.json')
    assert verified['status']=='independent_download_reconstruction_all_hashes_and_CPU_inference_passed'
    assert b['completed_optimizer_updates']==5000 and verified['full_validations']==20
    env=read(evidence/'environment.json');hw=read(evidence/'hardware.json');audit=read(evidence/'heads_audit.json')
    grad=read(evidence/'gradient_diagnostics.json');checkpoint=read(evidence/'checkpoint_audit.json')
    archive=read(evidence.parent/'artifact_manifest.json');cuda=read(evidence/'cuda_recovery/recovery.json')
    files=read(links);generation=read(evidence/'analysis/generation.json')
    assert all(cuda['exact_state'].values()) and cuda['additional_optimizer_updates']==0
    output.mkdir(parents=True,exist_ok=False);(output/'figures').mkdir();(output/'evidence').mkdir()
    for folder in [evidence/'analysis/figures',comparison/'figures']:
        for p in folder.glob('*'):
            if p.suffix in {'.png','.pdf'}:shutil.copyfile(p,output/'figures'/p.name)
    for name in ['summary.json','validation.csv','generation.json','generation.md']:shutil.copyfile(evidence/'analysis'/name,output/name)
    shutil.copyfile(comparison/'matched_validation.csv',output/'matched_validation.csv')
    for p in evidence.glob('*.json'):shutil.copyfile(p,output/'evidence'/p.name)
    shutil.copyfile(evidence/'cuda_recovery/recovery.json',output/'evidence/cuda_recovery.json')
    for name in ['artifact_manifest.json','archive_parts.json','independent_verification.json']:shutil.copyfile(evidence.parent/name,output/'evidence'/name)
    shutil.copyfile(comparison/'comparison.json',output/'evidence/comparison.json');shutil.copyfile(links,output/'evidence/artifact_links.json')
    notes=qualitative.read_text()
    text=f'''# E07 seed42 — eight-head attention experiment

Exactly **5,000 completed optimizer updates**, **20 full sliding validations**, **20 complete validation checkpoint states**, and **{checkpoint['snapshot_count']} verified immutable snapshots**. Best: step **{best['step']}**, loss **{best['loss']:.8f}**, conditional BPB **{best['bpb']:.8f}**. Final5000: loss **{final['loss']:.8f}**, BPB **{final['bpb']:.8f}**. Authentic fixed-protocol generations, independent archive integrity/inference and compatible CUDA full-state restoration passed. Single seed; no significance claim.

## 1. Question, hypothesis and scope

How does changing six32-dimensional attention heads to eight24-dimensional heads, at the same width192 and parameter count, affect next-token prediction, optimization, efficiency and generated text? More heads provide additional independently normalized attention distributions with smaller per-head spaces. Improvement, regression and negligible differences were possible hypotheses. No direction was assumed.

The authorized research scope is E07 seed42, exactly5000 updates, fresh initialization. No other research configurations, seeds, interactions or10k extension ran. Labeled smoke updates and recovery checks are separate engineering evidence. After Colab refused GPU allocation, the user explicitly requested Kaggle and completion on its verified account. This platform change is documented; scientific settings stayed frozen. No test likelihood or test-based selection.

## 2. Technical head-count distinction and correctness audit

The unchanged implementation projects Q/K/V from192 to192, splits to `(batch,8,length,24)`, computes `(Q @ K.T)/sqrt(24)`, masks future positions, softmaxes along key positions per head, applies training attention dropout0.1, aggregates V, merges to192 and applies the existing192-dimensional output projection. E01 instead splits to6×32. Projection tensor shapes and trainable parameter counts are identical. Head splitting changes the score distributions and attention/dropout tensor sizes; shared seed initialization does not force identical optimization trajectories.

CPU and qualified CUDA numerical audits inspected all four actual blocks. Manual sqrt24 probabilities/output matched; incorrect sqrt32 and sqrt192 references differed numerically. QKV/logit/merge shapes, per-head key softmax, future mask, train/eval dropout scaling, GELU MLP192→768→192, sinusoidal positions and absence of ReLU/RoPE/learned positions passed. All68 initial E01/E07 state tensors matched under seed42 in the same environment. CUDA initial parameter-byte SHA256: `{audit['initial_parameter_sha256']}`.

Shifted next-token targets, manual cross-entropy, finite forward/backward/all parameter gradients and nonzero gradients for every Q/K/V/output projection passed. Altering future tokens preserved earlier logits. One greedy generation token matched prediction from the last128-token cropped context. Parameter count exactly **1,804,800**. No scientific implementation fix was needed. [Full audit](evidence/heads_audit.json), [existing suite](evidence/tests.json), [focused attention test](evidence/attention_test.json). Test counts/times are recorded in those actual outputs.

## 3. Source, software and hardware

- Clean trained reviewed source: `{b['source_commit']}`.
- Scientific source SHA256: `{b['code_hash']}`.
- Actual resolved E07 config SHA256: `{b['config_hash']}`.
- E01 source: `{a['source_commit']}`; canonical source/config diff and historical audited hash establish equivalent shared model/tokenizer/training/evaluation semantics.
- Installed shared `nano_lm` remained at pinned clean main. Separately committed orchestration was executed from its own checkout; notebook contains exact orchestration SHA. No separate training loop/model implementation.
- [Research notebook]({files['notebook']}), [qualification notebook]({files['qualification_notebook']}).
- Isolated Python{env['python']}, PyTorch{env['torch']}, torch CUDA{env['cuda']}, NumPy1.26.4, tokenizers0.21.0, Matplotlib3.9.4, pytest8.3.4. Exact46-package qualified transitive freeze matched E01/E02/E05 apart from editable project location.
- Tesla T4; {hw['gpu_count']} visible devices, cuda:0 used for this one run. Per-device total memory {hw['devices'][0]['total_bytes']:,}bytes. Driver/hardware in [hardware receipt](evidence/hardware.json); full freeze/platform details in [environment](evidence/environment.json).

Kaggle native Python3.13/PyTorch2.11 was left unchanged and did not train/evaluate/generate the research model. `MPLBACKEND=Agg`, deterministic settings and OMP threads1 reproduced the qualified workflow. E01 also ran Kaggle/T4/driver580.178.04; software and accelerator matches reduce runtime confounds, while different host/session scheduling and thermal/load conditions remain uncontrolled. No isolated head-count speed mechanism or FLOP claim.

## 4. Dataset, tokenizer and likelihood policy

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e; raw SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; character SHA256 `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`. Train/validation/test1003854/55770/55770bytes; vocabulary65. All reference manifest fields verified. Splits precede train-only fitting; windows/shifted targets remain within their partition. [Data evidence](evidence/dataset_verification.json).

Every full sliding validation uses complete55770byte validation text,256conditioning-only bytes and exactly55514 scored target bytes/characters, each once, context128, batch32, eval dropout disabled and canonical local position reset. No cap/subsampling/test likelihood. Conditional BPB equals NLL/(ln2×55514) and character loss/ln2; perplexity=exp(loss). E01/E07 loss and BPB compare directly. Full-validation loss selects the best checkpoint.

## 5. Frozen model and optimizer

Decoder-only from-scratch GPT, character65, width192,4blocks,8heads/24dim, context128, MLPexpansion4 (192→768→192), GELU, sinusoidal, dropout0.1. Sole intended E01 change `n_heads:8`; params1804800.

Seed42, float32 CUDA, AdamW3e-4, constant LR, weight decay0.01, clipping1.0, microbatch4, accumulation8,4096targets/update.5000 completed updates; no AMP-skipped updates. Validation250/checkpoint500 with every validation state preserved. Full optimizer/scheduler/scaler/RNG/sampling/history state retained; no fine-tuning or dynamic hyperparameter changes. [Execution receipt](evidence/execution.json).

## 6. Genuine training and matched full validation curves

Recorded update1 train loss {b['initial_recorded_training']['loss']:.8f}; first full validation250 loss {b['initial_recorded_validation']['loss']:.8f}; final sampled train loss {b['final_training_loss']:.8f}; descriptive trailing100 mean {b['final_training_trailing_100_loss']:.8f}. No step-zero measurement invented.

| Update | E01 loss | E07 loss | E01 BPB | E07 BPB | E07−E01 loss |
|---:|---:|---:|---:|---:|---:|
'''
    for r in c['rows']:text+=f"| {r['step']} | {r['E01_loss']:.8f} | {r['E07_loss']:.8f} | {r['E01_bpb']:.8f} | {r['E07_bpb']:.8f} | {r['loss_difference_E07_minus_E01']:+.8f} |\n"
    text+='\n[Matched20-point CSV](matched_validation.csv), [full E07 validation/perplexity table](validation.csv).\n'
    for label,name in [('Training loss','training_loss'),('Full sliding validation loss','validation_loss'),('Conditional BPB','validation_bpb'),('Matched loss','e01_e07_validation_loss'),('Matched BPB','e01_e07_validation_bpb')]:text+=f'\n![{label}](figures/{name}.png)\n'
    text+='\nLines join measured validation endpoints. Smoothing is descriptive. Dropout-active sampled train loss is not full train-corpus likelihood; its gap with evaluation-mode validation alone cannot establish overfitting. One final fluctuation does not prove convergence or deterioration.\n\n## 7. Endpoint and efficiency comparison\n\n| Metric | E01 six heads/32dim | E07 eight heads/24dim |\n|---|---:|---:|\n'
    metrics=[('Parameters',a['parameters'],b['parameters']),('Best step',a['best_validation']['step'],best['step']),('Best loss',a['best_validation']['loss'],best['loss']),('Best BPB',a['best_validation']['bpb'],best['bpb']),('Final loss',a['final_validation']['loss'],final['loss']),('Final BPB',a['final_validation']['bpb'],final['bpb']),('Final perplexity',a['final_validation']['perplexity'],final['perplexity']),('Training seconds',a['training_seconds'],b['training_seconds']),('Full validation seconds',a['validation_seconds'],b['validation_seconds']),('Updates/sec',5000/a['training_seconds'],5000/b['training_seconds']),('Predicted tokens/sec',20480000/a['training_seconds'],20480000/b['training_seconds']),('Peak allocated MiB',a['peak_gpu_bytes']/2**20,b['peak_gpu_bytes']/2**20),('Clipping fraction',c['E01_clipping_frequency'],c['E07_clipping_frequency']),('Unique target bytes',a['unique_target_bytes'],b['unique_target_bytes'])]
    for label,x,y in metrics:text+=f'| {label} | {x:.8f} | {y:.8f} |\n'
    text+=f'''
Best loss/BPB reduction: {c['best_loss_reduction_percent']:+.4f}%; final reduction {c['final_loss_reduction_percent']:+.4f}%. Positive means lower E07; negative means regression. Training-time change {c['training_time_change_percent']:+.3f}%. Both runs exposed20480000predicted tokens/original target bytes. Exposure includes repeats; unique coverage is not independent sample count or complete epochs.

Loss decrease3k→5k: E01 {a['loss_improvement_3000_to_5000']:.8f}, E07 {b['loss_improvement_3000_to_5000']:.8f}. Descriptive late fitted loss slopes per1000updates: E01 {a['late_validation_slope_per_1000_steps']:.8f}, E07 {b['late_validation_slope_per_1000_steps']:.8f}; no prediction of longer-run behavior.

Measured local checkpoint overhead {c['checkpoint_seconds']:.3f}s; snapshot export into Kaggle saved-output staging {c['Drive_export_seconds']:.3f}s (historical comparison JSON key `Drive_export_seconds`; **these are Kaggle local exports**, not Drive upload latency); journal/fsync {c['journal_seconds']:.3f}s; engine wall {c['engine_wall_seconds']:.3f}s. Archive construction/local staged publication {archive['archive_and_publication_seconds']:.3f}s. Platform saved-version publication and later Drive uploads are distinct stages; only separately recorded timings are claimed. Peak allocated memory excludes reserved memory/driver overhead.

![Throughput](figures/throughput.png)
![Training time](figures/training_time.png)

## 8. Gradients and head-related observations

All5000 gradient norms finite. Mean{grad['norm_mean']:.6f}, median{grad['norm_median']:.6f}, min{grad['norm_min']:.6f}, max{grad['norm_max']:.6f}, final{grad['final_norm']:.6f}. {grad['steps_above_clipping_threshold']}updates exceeded1.0 ({100*grad['clipping_frequency']:.3f}%). E01 clipping0.140%. [Actual diagnostics](evidence/gradient_diagnostics.json).

![Gradient norm](figures/gradient_norm.png)
![Constant learning rate](figures/learning_rate.png)

Finite gradients and the preflight math audit establish observed numerical correctness; they do not identify a learned attention diversity mechanism. No systematic trained-head specialization, causal head ablation or representation-diversity audit was collected. Loss changes alone cannot establish those mechanisms.

## 9. Best/final authentic text generation

Unchanged generation protocol: five fixed prompts, greedy and temperatures0.7/1.0/1.2, seeds42/123/456,256new character tokens, best/final policies.100 unedited labels (50settings/policy), checkpoint hashes verified. [Full readable grid](generation.md), [full JSON](generation.json).
'''
    text+=('\nBest/final identical weights;50duplicate policy pairs are not independent evidence.\n' if best['step']==5000 else f"\nBest{best['step']} and final5000 are distinct weights; both full grids retained.\n")
    text+='\n'+notes+'\n\n### All five predetermined best greedy continuations\n'
    for s in generation['samples']:
        if s['checkpoint_policy']=='best' and s['greedy']:text+=f"\n#### {s['prompt']!r}\n\n```text\n{s['text']}\n```\n"
    text+='\n### Complete-grid repetition statistics\n\n| Experiment | Policy | Mode | Samples | Repeated8-gram fraction |\n|---|---|---|---:|---:|\n'
    for r in c['generation_repetition']:text+=f"| {r['experiment']} | {r['policy']} | {r['mode']} | {r['samples']} | {r['mean_repeated_8gram_position_fraction']:.6f} |\n"
    text+=f'''
Character8-grams counted over256-character continuations excluding prompts; repeated-position fraction is descriptive, not fluency/coherence quality. Higher-temperature novelty does not imply better text. Outputs were not rewritten or cherry-picked.

## 10. Durable storage and independent recovery

Colab failed allocation before any code/training. Kaggle's legitimate verified account provided T4 access. Before research, a real E07 smoke checkpoint was exported with manifests/hash checks, saved as a private persistent Kaggle output version, independently downloaded/verified and published/read back from Drive. A fresh compatible isolated CUDA process restored complete model/AdamW/scheduler/scaler/Python/NumPy/torch/CUDA/sampling RNG/training state from the mounted saved smoke archive and compared an exact state roundtrip. Working-directory copies alone were not treated as persistent storage. Final artifacts use the shared verified snapshot workflow and staged-rename publisher; no Drive FUSE hard links assumed.

Research run `{b['run_id']}`.20checkpoints and{checkpoint['snapshot_count']}snapshots preserved. Each checkpoint's optimizer counters/scheduler epoch/history length/metadata verified. Research failure/resume events: {len(checkpoint['journal_failures_or_resumes'])}. [Inventory](evidence/checkpoint_audit.json), [snapshot index](evidence/snapshot_index.json).

Final compatible CUDA process restored all six complete state structures exactly; best/final inference passed with zero additional updates. No10k continuation executed. [CUDA receipt](evidence/cuda_recovery.json).

- [Persistent private Drive folder]({files['folder']}).
- [Saved Kaggle research output]({files['notebook']}).
- Complete archive size **{archive['archive_bytes']:,}bytes**; SHA256 `{archive['archive_sha256']}`.
- [File-hash manifest](evidence/artifact_manifest.json), [numbered parts](evidence/archive_parts.json), [independent integrity/inference receipt](evidence/independent_verification.json).

Independent retrieval/reconstruction verified full archive hash/size,{verified['archive_entries']}safe unique entries/CRC/per-file hashes, all20actual checkpoint state inventories,5000history rows/20full validations and all100generation checkpoint hashes/protocol settings. Best/final CPU inference succeeded; CPU migration profile Python{verified['CPU_environment']['python']}/torch{verified['CPU_environment']['torch']} is not the exact training-resume environment. Fresh CUDA recovery used the approved profile. Large weights/datasets remain outside GitHub.

## 11. Failures, uncertainty and limitations

Colab GPU usage limits blocked its initial attempt with zero research/smoke updates. User authorized Kaggle fallback and its profile was independently qualified. All actual additional failures and timing limitations are retained in the evidence; incomplete transfers/checks are never counted as successful preservation. One seed offers no variance/significance estimate. Same initialization and source/effective budget improve control, but changing head count changes attention shapes/dropout RNG consumption and optimization trajectory. Same platform/GPU/software does not fully control host/runtime conditions. Validation is conditional sliding likelihood, not whole-file entropy. Repetition metrics and human qualitative assessment do not measure narrative quality formally. No test-based tuning, other trained configuration/seed or extension.

## 12. Interpretation, optimization implications and recommendations

{files['interpretation']}

Additional independently authorized seeds are needed to estimate variance and decide whether head count should change in the final architecture. Longer training would require a separately approved continuation from verified5000state, preserving constant schedule/optimizer/RNG. No combined configuration is authorized by this result. Learned attention diversity requires targeted diagnostics before a mechanism claim.

## 13. E06/E01/E07 and collaboration

{files['E06_status']}

Dedicated E07 branch, private Kaggle notebooks, Drive namespace and report. Primary researcher owns E06; coordination on PR9 records separate scopes. E01–E06 evidence/scientific files unchanged. [Review PR]({files['pr']}) contains reproducible scripts/launcher and exact executed notebook snapshots, tests, small evidence, tables, figures and detailed documentation. No merge.
'''
    (output/'E07_seed42_analysis.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','evidence','comparison','output','links','qualitative']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.root,a.evidence,a.comparison,a.output,a.links,a.qualitative)
