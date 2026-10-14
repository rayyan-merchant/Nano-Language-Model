"""Acceptance-gated B01 variable-stop report, fixed grids and real comparisons."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nano_lm.artifacts import atomic_json, file_hash
from scripts.pilot.diagnostics import longest_overlap
from scripts.final.report import savefig
from scripts.balanced.protocol import MODEL, PARAMETERS, STOP_POLICY


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def generation_diagnostics(root,grid,data):
    source=(data/'train.txt').read_bytes();training=source.decode('utf-8')
    grids={'B01':grid,'E02':read(root/'report/pilots/E02_seed42/generation.json'),
           'F01':read(root/'report/final/generation.json')}
    rows=[];groups=[];matched={}
    for experiment,g in grids.items():
        assert g['protocol']==grid['protocol'] and len(g['samples'])==100
        seen=set();buckets={}
        for slot in g['samples']:
            key=(slot['checkpoint_policy'],slot['prompt'],slot['greedy'],slot['temperature'],slot['seed'])
            assert key not in seen and slot['max_new_tokens']==256;seen.add(key)
            assert slot['text'].startswith(slot['prompt'])
            text=slot['text'][len(slot['prompt']):];words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text)
            measures={}
            for n in [3,8,24,48]:
                grams=[text[i:i+n] for i in range(max(0,len(text)-n+1))]
                measures[f'repeated_{n}gram_fraction']=1-len(set(grams))/len(grams) if grams else None
            pairs=list(zip(words,words[1:]));word_diversity=len(set(pairs))/len(pairs) if pairs else None
            row=dict(experiment=experiment,checkpoint_policy=slot['checkpoint_policy'],prompt=slot['prompt'],
                greedy=slot['greedy'],temperature=slot['temperature'],seed=slot['seed'],
                checkpoint_sha256=slot['checkpoint_sha256'],continuation_characters=len(text),
                replacement_characters=text.count('\ufffd'),distinct_word_bigram_fraction=word_diversity,
                speaker_labels=re.findall(r"(?m)^([A-Za-z][A-Za-z '\-]{1,32}):\s*$",text),**measures)
            if experiment=='B01':row.update(longest_overlap(text,training))
            rows.append(row);matched.setdefault(repr(key),{})[experiment]={'text':slot['text'],'descriptors':row}
            buckets.setdefault((slot['checkpoint_policy'],slot['greedy'],slot['temperature']),[]).append(row)
        for key,values in buckets.items():
            groups.append(dict(experiment=experiment,checkpoint_policy=key[0],greedy=key[1],temperature=key[2],
                outputs=len(values),mean_continuation_characters=statistics.mean(r['continuation_characters'] for r in values),
                mean_repeated_8gram_fraction=statistics.mean(r['repeated_8gram_fraction'] for r in values),
                replacement_characters=sum(r['replacement_characters'] for r in values)))
    assert len(matched)==100 and all(set(v)==set(grids) for v in matched.values())
    return {'rows':rows,'groups':groups,'all100_matched_settings':matched,'training_sha256':file_hash(data/'train.txt'),
        'scope':'All300 genuine fixed slots; all100 B01 train-only exact overlaps; unedited outputs. String diversity/labels/novelty are not grammar or semantic coherence scores; duplicated policy outputs are not independent samples.'}


def write(root,verified,manifest_path,cuda_path,out,kernel,data):
    root,verified,out,data=map(Path,[root,verified,out,data]);out.mkdir(parents=True,exist_ok=False)
    m=read(manifest_path);cpu=read(verified/'independent_recovery.json');cuda=read(cuda_path)
    c=read(verified/'completion.json');stop=read(verified/'stopping_summary.json');s=read(verified/'analysis/summary.json')
    assert cpu['authoritative_updates']==cuda['completed_updates']==s['completed_optimizer_updates']==stop['actual_completed_updates']
    target=stop['actual_completed_updates'];assert len(cpu['checkpoint_counters'])==target//250
    assert cpu['archive_sha256']==cuda['archive_sha256']==m['archive_sha256']
    assert cuda['status']=='B01_published_research_fresh_CUDA_full_state_and_fixed_grid_passed'
    assert all(cuda['exact_state'].values()) and cuda['additional_optimizer_updates']==0
    assert cuda['verified_snapshot_count']==2*(target//250)+1
    assert cuda['generation_reproduction']['slots']==100 and cuda['generation_reproduction']['additional_optimizer_updates']==0
    reproduction_path=Path(cuda_path).parent/'generation_reproduction/generation_reproduction.json'
    reproduced=read(reproduction_path)
    assert reproduced['status']=='all100_B01_fixed_generations_exactly_reproduced'
    assert len(reproduced['samples'])==100 and reproduced['additional_optimizer_updates']==0
    assert all(r['original_generation_exactly_reproduced'] and len(r['generated_token_ids'])==256 for r in reproduced['samples'])
    assert stop['policy']==STOP_POLICY and stop['test_likelihood_evaluations']==0
    run=(verified/c['final_checkpoint']).parent.parent;metadata=read(run/'metadata.json')
    assert metadata['parameters']==PARAMETERS and metadata['resolved']['model']==MODEL
    assert metadata['resolved']['tokenizer_hash']=='2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55'
    evidence=out/'evidence/research';evidence.mkdir(parents=True)
    for name in ['completion.json','stopping_summary.json','source_provenance.json','dataset_validation.json','environment.json',
                 'hardware.json','CUDA_architecture_audit.json','development_recovery_gate.json','execution_plan.json',
                 'snapshot_index.json','wall_time.json','io_timing.json','independent_recovery.json']:
        shutil.copyfile(verified/name,evidence/name)
    shutil.copyfile(manifest_path,evidence/'artifact_manifest.json');shutil.copyfile(cuda_path,evidence/'CUDA_recovery.json')
    shutil.copyfile(reproduction_path,evidence/'CUDA_generation_reproduction.json')
    shutil.copyfile(verified.parent/'download_receipt.json',evidence/'download_receipt.json')
    shutil.copyfile(run/'metadata.json',evidence/'run_metadata.json');shutil.copyfile(run/'dataset_manifest.json',evidence/'dataset_manifest.json')
    for name in ['summary.json','validation.csv','generation.json','generation.md','generation_token_diagnostics.json','gradient_intervals.json']:
        shutil.copyfile(verified/'analysis'/name,out/name)
    shutil.copytree(verified/'analysis/figures',out/'figures')
    grid=read(out/'generation.json');quality=generation_diagnostics(root,grid,data)
    atomic_json(out/'generation_comparison.json',quality)
    histories=read(verified/'analysis/measured_history.json')['training'];val=[r for r in histories if 'validation' in r]
    assert len(histories)==target and [r['step'] for r in val]==list(range(250,target+1,250))
    norms=np.array([r['gradient_norm'] for r in histories])
    references={'E02':read(root/'report/pilots/E02_seed42/summary.json'),'F01':read(root/'report/final/summary.json')}
    rows=[]
    for eid,summary in [*references.items(),('B01',s)]:
        rows.append(dict(experiment=eid,seed=42,replicates=1,standard_deviation=None,parameters=summary['parameters'],
            updates=summary['completed_optimizer_updates'],best_step=summary['best_validation']['step'],
            best_BPB=summary['best_validation']['bpb'],final_BPB=summary['final_validation']['bpb'],
            training_seconds=summary['training_seconds'],validation_seconds=summary['validation_seconds'],
            peak_GPU_MiB=summary['peak_gpu_bytes']/2**20,training_source_bytes=summary['training_source_bytes'],
            best_relative_BPB_percent_vs_E02=100*(summary['best_validation']['bpb']/references['E02']['best_validation']['bpb']-1)))
    atomic_json(out/'E02_F01_B01_comparison.json',{'runs':rows,'scope':'Different combined architectures, budgets, checkpoint-selection opportunities and sessions; no replicated significance estimate'})
    with (out/'E02_F01_B01_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures=out/'comparison_figures';figures.mkdir()
    fig,ax=plt.subplots(figsize=(9,5))
    matched_validation={}
    for eid,path in [('E02',root/'report/pilots/E02_seed42/validation.csv'),('F01',root/'report/final/validation.csv'),('B01',out/'validation.csv')]:
        measurements=list(csv.DictReader(path.open(encoding='utf-8')))
        matched_validation[eid]={int(r['step']):r for r in measurements}
        ax.plot([int(r['step']) for r in measurements],[float(r['bpb']) for r in measurements],marker='.',label=eid)
    matched_rows=[]
    for step in range(250,target+1,250):
        row={'step':step}
        for eid in ['E02','F01','B01']:
            measured=matched_validation[eid][step]
            row.update({f'{eid}_{metric}':float(measured[metric]) for metric in ['loss','bpb','perplexity']})
        row['B01_minus_E02_BPB']=row['B01_bpb']-row['E02_bpb']
        row['B01_minus_F01_BPB']=row['B01_bpb']-row['F01_bpb']
        matched_rows.append(row)
    with (out/'matched_validation_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(matched_rows[0]));w.writeheader();w.writerows(matched_rows)
    ax.set(xlabel='Completed optimizer updates',ylabel='Conditional validation BPB',title='Recorded trajectories; B01 stops at frozen rule, no extrapolation')
    ax.legend();ax.grid(alpha=.2);savefig(fig,figures,'validation_BPB_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for row in rows:
        axes[0].scatter(row['parameters']/1e6,row['best_BPB']);axes[0].annotate(row['experiment'],(row['parameters']/1e6,row['best_BPB']),xytext=(5,5),textcoords='offset points')
    axes[0].set(xlabel='Parameters (millions)',ylabel='Selected validation BPB',title='Single-seed capacity/quality trade-off')
    axes[1].bar([r['experiment'] for r in rows],[r['peak_GPU_MiB'] for r in rows]);axes[1].set(ylabel='Peak allocated GPU MiB',title='Canonical training telemetry scope')
    savefig(fig,figures,'parameters_quality_memory')
    fig,ax=plt.subplots(figsize=(8,4.5));greedy=[r for r in quality['groups'] if r['greedy'] and r['checkpoint_policy']=='best']
    ax.bar([r['experiment'] for r in greedy],[r['mean_repeated_8gram_fraction'] for r in greedy])
    ax.set(ylabel='Repeated 8-character-gram fraction',title='All five fixed greedy prompts; BPE lengths still vary')
    savefig(fig,figures,'fixed_greedy_repetition')
    best=s['best_validation'];final=s['final_validation'];bpb2=references['E02']['best_validation']['bpb'];bpbF=references['F01']['best_validation']['bpb']
    token_diagnostics=read(out/'generation_token_diagnostics.json')
    assert reproduced['generation_file_sha256']==file_hash(out/'generation.json')
    assert all(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(token_diagnostics['samples'],reproduced['samples']))
    inference=[]
    for policy in ['best','final']:
        request=[r for r in token_diagnostics['samples'] if r['checkpoint_policy']==policy]
        assert len(request)==50
        inference.append({'policy':policy,'requests':50,'first_cold_request_seconds':request[0]['seconds'],
            'remaining49_mean_seconds':statistics.mean(r['seconds'] for r in request[1:]),
            'remaining49_generated_BPE_tokens_per_second':49*256/sum(r['seconds'] for r in request[1:])})
    duplicate=file_hash(verified/c['best_checkpoint'])==file_hash(verified/c['final_checkpoint'])
    if duplicate:
        for a,b in zip([r for r in grid['samples'] if r['checkpoint_policy']=='best'],[r for r in grid['samples'] if r['checkpoint_policy']=='final']):assert a['text']==b['text']
    findings={'best':best,'final':final,'actual_updates':target,'actual_validations':target//250,'stopping_reason':stop['stopping_reason'],
        'gradient_norm':{'median':float(np.median(norms)),'p95':float(np.quantile(norms,.95)),'maximum':float(norms.max()),
                         'clipped_updates':int((norms>1).sum()),'clipped_fraction':float((norms>1).mean())},
        'inference':inference,'model_CPU_loads':token_diagnostics['loads'],'duplicate_best_final_weights':duplicate,
        'strict_UTF8_invalid_outputs':token_diagnostics['strict_UTF8_invalid_outputs'],
        'best_absolute_BPB_difference_vs_E02':best['bpb']-bpb2,'best_absolute_BPB_difference_vs_F01':best['bpb']-bpbF,
        'source_byte_sampling':cpu['sampler'],'original_test_evaluations':0}
    atomic_json(out/'findings.json',findings)
    table='\n'.join(f"| {r['step']} | {r['validation']['loss']:.8f} | {r['validation']['bpb']:.8f} | {r['validation']['perplexity']:.6f} |" for r in val)
    stop_table='\n'.join(f"| {r['step']} | {r['bpb']:.8f} | {r['patience_reference_after']:.8f} | {r['counter']} | {r['meaningful_improvement']} | {r['stopping_reason'] or ''} |" for r in stop['patience_history'])
    compare_table='\n'.join(f"| {r['experiment']} | {r['parameters']:,} | {r['updates']} | {r['best_step']} | {r['best_BPB']:.8f} | {r['final_BPB']:.8f} | {r['training_seconds']:.3f} | {r['peak_GPU_MiB']:.3f} |" for r in rows)
    proof=read(evidence/'source_provenance.json');wall=read(evidence/'wall_time.json')
    report=f'''# B01 — NanoLM-Balanced seed 42: measured research analysis

## Objective, motivation and frozen hypothesis

Can a smaller BPE/RoPE model retain competitive held-out prediction and genuine text quality while avoiding F01's severe late deterioration? Capacity, repeated source exposure, constant LR, optimization and component interactions are hypotheses, not isolated causes of F01's failure. B01 combines six blocks/RoPE with original width192; it is not a single-factor ablation or an optimal architecture claim.

## Exact architecture and provenance

```json
{json.dumps(MODEL,indent=2)}
```

Exactly {PARAMETERS:,} parameters: embeddings196,608 + six blocks2,669,184 + final LN384 + untied head196,608. Head32, MLP192→768→192/GELU/preLN, dropout0.1; Q/K RoPE only, scaled attention sqrt32, causal masking and crop128. No pretrained weights/black-box Transformer. Numerical CPU and actual qualified CUDA references check wrong scale/sign, future invariance, gradients, shifted manual CE, norms, relative position and BPE byte coverage.

Trained source `{s['source_commit']}`; scientific fingerprint `{proof['scientific_core_fingerprint']}`; aggregate `{s['code_hash']}`; resolved config `{s['config_hash']}`. Original implementation/E01–E14/F01 evidence remains unchanged. The B01 JSON changes the aggregate, not scientific semantics. Exact dependency/hardware records are in evidence/research; isolated Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/T4 device0/FP32.

## Data and method

Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed; BPE checksum2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55. Original bytes1,003,854/55,770/55,770. Canonical tokenizer artifacts were reused without refitting. Shared acquisition/splits/manifest validation preserve no cross-boundary sampling and train-only fit lineage. No original test likelihood or external dataset evaluation occurred.

AdamW constant3e-4, weight decay0.01, clipping1, float32, microbatch4/accumulation8,32 sampled windows/4,096 targets per optimizer update, maximum5,000. Full sliding validation every250 scores55,514 original bytes/24,386 BPE targets after256-byte conditioning-only exclusion, dropout off, context128BPE, batch32. BPB=NLL/(ln2×55,514), a boundary/context-constrained conditional metric.

## Frozen stopping and complete observations

Reference initializes at250; reset only if bpb < reference−0.001, strict unrounded double precision. Otherwise counter increments, including before eligibility2,000. Stop at first eligible counter>=6 or budget5,000. Absolute minimum selects the best without tolerance. Boundary/nextafter, cumulative tiny improvement, eligibility and stop-after-decision regression tests precede research.

Actual {target:,} authoritative and known physical updates, {target//250} full validations, stopping reason **{stop['stopping_reason']}**. Planned resumes={target//250-1}; they are not failures or replay. No updates followed the decision. No unobserved later points are invented. Full patience receipt is preserved.

| Update | Loss (nats/token) | Conditional BPB | Token perplexity |
|---:|---:|---:|---:|
{table}

| Update | BPB | Patience reference | Counter | Meaningful reset | Stop reason |
|---:|---:|---:|---:|---|---|
{stop_table}

Best update{best['step']}: loss{best['loss']:.8f}, BPB{best['bpb']:.8f}, perplexity{best['perplexity']:.6f}. Final update{target}: loss{final['loss']:.8f}, BPB{final['bpb']:.8f}, perplexity{final['perplexity']:.6f}. Final training loss{s['final_training_loss']:.8f}; trailing100 mean{s['final_training_trailing_100_loss']:.8f}. First training step1 and validation250 are recorded; no fake step-zero measurement.

![Actual loss](figures/combined_loss.png)
![Matched BPB](comparison_figures/validation_BPB_comparison.png)

## Exact segmented execution and persistence preflight

Original nano_lm.train CLI runs complete250-update segments with the same run/config/checkpoint. It reconstructs objects, restores saved optimizer/scheduler/scaler/RNG/sampler, and never trains newly initialized weights after a resume. Preflight uninterrupted4-update versus2+2 smoke trajectories match all five optimization/RNG fields and every nontelemetry nested training/history field exactly, including full sliding CUDA losses. Wall times/throughput/allocator peaks are measured separately and cannot be equal clocks; raw inequality is disclosed. Zero-update raw roundtrip matches all six fields including full telemetry/history. Published development was independently downloaded/CRC/hash/CPU verified before fresh compatible CUDA restore and smoke continuation; research only launched after those gates.

## Resources, exposure and gradients

Training-only{s['training_seconds']:.3f}s; full validation{s['validation_seconds']:.3f}s; {target/s['training_seconds']:.6f}updates/s and{s['training_tokens']/s['training_seconds']:.3f}BPEtargets/s. Wrapper before archive{wall['pipeline_seconds_before_archive']:.3f}s; archive/hash/CRC{m['archive_and_CRC_seconds']:.3f}s. Inner train/validation/IO scopes overlap wrapper and must not be double-counted; bootstrap/queue/publication/retrieval are separate. Peak{s['peak_gpu_bytes']/2**20:.3f}MiB is sampled training allocator telemetry, not all device/driver memory. Each planned segment resets the peak counter and ends at validation; validation/inference peaks are not separately recorded. F01 uninterrupted telemetry could include prior validation peaks, so this memory comparison has a measurement-scope confound.

Actual predicted targets{s['training_tokens']:,}; original target-byte exposure{s['training_source_bytes']:,}; unique target bytes{s['unique_target_bytes']:,}. Independent sampler RNG/union replay passes. These are repeated target exposure, not complete epochs/FLOPs. Inference latency uses only the actual fixed grid, synchronized before/after original generation; first request per policy is cold and remaining49 statistics are in findings.json. CPU checkpoint-deserialize/model-build time excludes CUDA transfer and is cache-dependent. Comparable E02/F01 per-request latency was not recorded; no invented benchmark substitutes for it.

Preclip norm median{findings['gradient_norm']['median']:.6f}, p95{findings['gradient_norm']['p95']:.6f}, max{findings['gradient_norm']['maximum']:.6f}; {findings['gradient_norm']['clipped_updates']}/{target} exceeded1.0. Every250-update interval is in gradient_intervals.json; no clipping-caused-overfitting or longitudinal per-layer gradient claim is inferred.

![Observed gradient norms](figures/gradient_norm.png)

## E02/F01/B01 measured comparison

| Model | Parameters | Updates | Best update | Best BPB | Final BPB | Training seconds | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{compare_table}

B01 selected-best difference versus E02={best['bpb']-bpb2:+.8f}BPB ({100*(best['bpb']/bpb2-1):+.3f}%); versus F01-best={best['bpb']-bpbF:+.8f}BPB. Size is49.3467% lower than F01 and40.9436% above E02. Comparisons vary depth/position/capacity/initialization, stopping/budgets and selection opportunities. The relevant F01 reference is its early selected1,750, not its worse final10,000. Equal BPE/context aligns token/byte coverage but does not imply equal compute. No statistical significance or cross-seed SD.

![Capacity and memory](comparison_figures/parameters_quality_memory.png)

## Genuine fixed generation and limitations

All100 best/final slots are preserved in generation.json/md: five prompts, greedy, temperatures0.7/1/1.2, sampling seeds42/123/456 and256 new BPE tokens. No repair, filtering, external-model rewriting, top-k/top-p or repetition penalty. Duplicate best/final weights={duplicate}; identical policy outputs are not independent samples. All300 matched E02/F01/B01 strings/descriptors remain unedited in generation_comparison.json. All100 B01 train-only longest literal substrings exclude prompt and record source offsets. Repetition/word diversity/speaker labels do not measure grammar or understanding; manual observations follow in B01_model_selection.md. Raw token/byte diagnostics found{token_diagnostics['strict_UTF8_invalid_outputs']} invalid-UTF8 outputs, retained without correction. The recovered fixed grid reproduces exactly.

![Fixed greedy repetition](comparison_figures/fixed_greedy_repetition.png)

## Durable publication and independent recovery

Research https://www.kaggle.com/code/{kernel}, saved version1. Archive{m['archive_bytes']:,}bytes, SHA256 `{m['archive_sha256']}`. Independent CRC/safe paths/exact{cpu['verified_file_count']}-file inventory/allSHA256/allactual optimizer/scheduler/history counters, full stopping-rule replay and CPU best/final inference pass. Native{2*(target//250)+1} immutable snapshots verified in fresh qualified CUDA; complete model/optimizer/scheduler/scaler/allRNG/private sampling/history preserved at{target}→{target}, zero additional research updates. All100 fixed strings reproduce. Full checkpoints/journals/archives remain on D: and legitimate Kaggle saved outputs, outside Git. Detailed post-recovery report/figures are lightweight Git deliverables. No original test assessment was performed.

## Interpretation and remaining uncertainty

Early stopping is a predefined heuristic, not proven optimal, and changes the observation horizon. The unexecuted future is unknown; B01 cannot prove it would never deteriorate beyond its stop. Repeated validation-based architecture/checkpoint choices, one contiguous corpus/seed, unequal budgets and different sessions constrain claims. B01's compactness alone does not beat E02. Separate prediction, compactness and real text behavior; see B01_model_selection.md for observations, interpretations and negative results. No further model/seed/extension is authorized.
'''
    # Preserve generated grids separately; this report contains no copied output fence.
    from scripts.final.report import readable_prose
    (out/'B01_seed42_analysis.md').write_text(readable_prose(report),encoding='utf-8')
    verdict='B01 has a lower observed validation minimum than E02.' if best['bpb']<bpb2 else 'E02 retains the lower observed validation minimum with fewer parameters; B01 has not established a predictive-quality justification for its extra capacity.'
    (out/'B01_model_selection.md').write_text(f'''# B01 model selection — measured comparison

**Observation:** {verdict} Difference{best['bpb']-bpb2:+.8f}BPB. F01's selected early minimum is{bpbF:.8f}; B01 differs by{best['bpb']-bpbF:+.8f}. B01 is49.3467% smaller than F01 but40.9436% larger than E02.

**Observation:** actual stop{target}, reason{stop['stopping_reason']}, best{best['step']}; best/final BPB{best['bpb']:.8f}/{final['bpb']:.8f}. Original F01's late10k degeneration is not a favorable baseline substitute. B01's counterfactual curve after stopping was not measured. Reduced repetition alone does not establish grammar/dialogue continuity; manual full-grid observations must be added before final acceptance of this narrative.

**Interpretation boundary:** one seed supplies no variance/significance estimate. Multiple architecture choices and repeated validation exposure, different budgets and model shapes confound a global optimum or isolated early-stop effect. No original test likelihood was evaluated for B01; the already exposed F01 test result did not tune B01.

**Proposed independent assessment, not executed:** use a curator-selected held-out set of complete public-domain plays demonstrably absent from the entire Tiny Shakespeare source, with provenance/overlap checks and source-byte scoring rules frozen before inspection. Freeze E02/F01/B01 checkpoint identities and tokenizer/context/prefix/batching beforehand; forbid tuning after this assessment. Independent custody and near-duplicate review are needed. No external corpus was downloaded or evaluated during this task.
''',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','verified','manifest','cuda-receipt','output','dataset']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kernel',required=True);a=p.parse_args()
    write(a.root,a.verified,a.manifest,a.cuda_receipt,a.output,a.kernel,a.dataset)
