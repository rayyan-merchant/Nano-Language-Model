"""Acceptance-gated final report and publication figures from genuine evidence."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.config import load_experiment
from scripts.final.protocol import MODEL, PARAMETERS, parameter_breakdown
from scripts.final.generation_diagnostics import describe


def readable_prose(text):
    """Format report prose without changing fenced evidence, links or hashes."""
    from scripts.pilot.write_e14_report import prose_spacing
    text=prose_spacing(text);result=[];fence=None
    replacements={'full validations':'full validations','all40':'all 40','All40':'All 40',
        'all14':'all 14','all100':'all 100','every100':'every 100','all six':'all six',
        'bestBPB':'best BPB','finalBPB':'final BPB','bestCP':'best CP','step5000':'step 5000',
        'step10000':'step 10000','atstep':'at step ','bestgreedy':'best greedy',
        'first2':'first 2','first256':'first 256','head40':'head 40','source06291e1':'source 06291e1',
        'launcher9a93b7a':'launcher 9a93b7a','wrapper10000':'wrapper 10000',
        'before_archive':'before_archive','bytes/24386':'bytes / 24386',
        'updates/s':'updates/s','microbatch4':'microbatch 4','accumulation8':'accumulation 8'}
    for line in text.splitlines(keepends=True):
        opening=re.match(r'^\s*(`{3,}|~{3,})',line)
        if opening:
            if fence is None:fence=opening.group(1)
            elif opening.group(1)[0]==fence[0] and len(opening.group(1))>=len(fence):fence=None
            result.append(line);continue
        if fence is None:
            # All literal hashes, inline code and URLs remain byte-for-byte.
            parts=re.split(r'(`[^`]*`|!?\[[^\]]*\]\([^)]*\)|https?://\S+|[A-Za-z0-9_./-]+\.(?:md|json|png|pdf|csv|pt)|[0-9a-f]{64}(?![0-9a-f])|\b[0-9a-f]{40}\b)',line)
            for i in range(0,len(parts),2):
                value=parts[i]
                for old,new in replacements.items():value=value.replace(old,new)
                value=re.sub(r'([A-Za-z])(\d)',r'\1 \2',value)
                value=re.sub(r'(\d)([A-Za-z])',r'\1 \2',value)
                # Restore registry identifiers and natural scientific notation.
                value=re.sub(r'\b([EF]) (\d{2})\b',r'\1\2',value)
                value=value.replace('3 e-4','3e-4').replace('4 x','4×')
                parts[i]=value
            line=''.join(parts)
        result.append(line)
    return ''.join(result)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def savefig(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=240)
    fig.savefig(folder/f'{name}.pdf');plt.close(fig)


def architecture(folder):
    fig,ax=plt.subplots(figsize=(12,7));ax.set_xlim(0,12);ax.set_ylim(0,8);ax.axis('off')
    boxes=[(0.4,6.3,3,1.0,'Canonical Byte-BPE\n1024 tokens; train-only'),
           (4.1,6.3,3.2,1.0,'Untied token embedding\n1024 × 240; dropout 0.1'),
           (8.1,6.3,3.4,1.0,'Eight independent blocks\nwidth 240; pre-LayerNorm'),
           (0.4,3.5,5.1,1.8,'Each block: LN → Q/K/V projections\n6 heads × 40 coordinates\nRoPE(Q,K); V unchanged\nQKᵀ / √40 → causal mask\nsoftmax → dropout 0.1 → V'),
           (6.4,3.5,5.1,1.8,'Merge heads → 240 → dropout → residual\nLN → Linear 240→960 → GELU\nLinear 960→240 → dropout → residual\nDropout 0.1; eight repetitions'),
           (2.3,1.3,3.2,1.0,'Final LayerNorm\n480 trainable parameters'),
           (6.4,1.3,4.1,1.0,'Untied LM head 240→1024\nShifted next-token cross-entropy')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor='#eef3f7',edgecolor='#34536d'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    ax.plot([9.8,9.8,3.],[6.1,5.7,5.7],color='#34536d',lw=1.8)
    ax.plot([9.,9.,3.9],[3.3,2.7,2.7],color='#34536d',lw=1.8)
    for a,b in [((3.5,6.8),(3.9,6.8)),((7.4,6.8),(7.9,6.8)),((3.,5.7),(3.,5.45)),((5.6,4.4),(6.2,4.4)),((3.9,2.7),(3.9,2.45)),((5.65,1.8),(6.25,1.8))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#34536d',lw=1.8))
    ax.text(6,.3,'6,046,560 parameters • context128 BPE tokens • FP32 • no additive position embedding in RoPE mode',ha='center',fontsize=10)
    ax.set_title('F01: actual frozen implementation and tensor dimensions',fontsize=14)
    savefig(fig,folder,'F01_architecture')
    fig,ax=plt.subplots(figsize=(6,5));angles=np.linspace(0,2*np.pi,200)
    ax.plot(np.cos(angles),np.sin(angles),color='#c9d3de')
    ax.arrow(0,0,1,0,width=.012,color='#34536d',length_includes_head=True)
    ax.arrow(0,0,np.cos(1),np.sin(1),width=.012,color='#b95436',length_includes_head=True)
    ax.text(.5,-.13,'(1,0): pair input',ha='center');ax.text(.45,.92,'R(1)·(1,0)',ha='center')
    ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),xlabel='even coordinate',ylabel='odd coordinate',
           title='RoPE reference: position1, pair0, head dimension40')
    ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.text(.5,.02,'θ(m,i)=m·10000^(−2i/40); Q/K adjacent pairs rotate; V unchanged\nMathematical basis-vector illustration; not a trained activation',ha='center',fontsize=9)
    savefig(fig,folder,'RoPE_pair_reference')


def write(root,verified,manifest_path,cuda_receipt,generation_receipt,out,kernel):
    root,verified,out=Path(root),Path(verified),Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=load(manifest_path);cpu=load(verified/'independent_recovery.json');cuda=load(cuda_receipt)
    transfer=load(verified.parent/'download_receipt.json')
    assert transfer['status']=='authorized_download_size_SHA256_verified'
    assert transfer['archive_sha256']==manifest['archive_sha256'] and transfer['archive_bytes']==manifest['archive_bytes']
    reproduced=load(generation_receipt)
    assert reproduced['status']=='all100_fixed_CUDA_generations_exactly_reproduced_with_token_and_byte_diagnostics'
    assert len(reproduced['samples'])==100 and reproduced['additional_optimizer_updates']==0
    assert all(s['original_generation_exactly_reproduced'] and len(s['generated_token_ids'])==256 for s in reproduced['samples'])
    assert reproduced['generation_file_sha256']==file_hash(verified/'analysis/generation.json')
    completion=load(verified/'completion.json');summary=load(verified/'analysis/summary.json')
    provenance=load(verified/'source_provenance.json');environment=load(verified/'environment.json')
    test=load(verified/'reserved_test_evaluation.json');freeze=load(verified/'selection_freeze.json')
    history=load(verified/'analysis/measured_history.json')['training'];validations=[r for r in history if 'validation' in r]
    assert completion['parameters']==PARAMETERS and summary['completed_optimizer_updates']==len(history)==10000
    assert len(validations)==40 and [r['step'] for r in validations]==list(range(250,10001,250))
    assert cpu['authoritative_updates']==10000 and cpu['full_validations']==40 and len(cpu['checkpoint_counters'])==40
    assert cpu['archive_sha256']==cuda['archive_sha256']==manifest['archive_sha256']
    assert cuda['status']=='published_F01_research_fresh_CUDA_full_state_passed_no_updates'
    assert cuda['step_before']==cuda['step_after']==10000 and cuda['additional_optimizer_updates']==0
    assert all(cuda['exact_state'].values()) and cuda['verified_snapshot_count']==41
    assert test['checkpoint_sha256']==freeze['checkpoint_sha256'] and test['selection_frozen_before_evaluation']
    assert summary['best_validation']['step']==freeze['best_step'] and test['metrics']['scored_bytes']==55514
    assert summary['code_hash']==environment['code_hash']==provenance['F01_aggregate_code_hash']
    source_run=(verified/completion['final_checkpoint']).parent.parent;metadata=load(source_run/'metadata.json')
    assert metadata['resolved']['model']==MODEL and metadata['parameters']==PARAMETERS
    # Preserve small native evidence byte-for-byte; full history/binaries stay external.
    evidence=out/'evidence/research';evidence.mkdir(parents=True,exist_ok=True)
    for name in ['completion.json','environment.json','source_provenance.json','dataset_validation.json','hardware.json',
                 'CUDA_architecture_audit.json','selection_freeze.json','reserved_test_evaluation.json','io_timing.json',
                 'wall_time.json','snapshot_index.json','execution_command.json','independent_development_recovery_gate.json','independent_recovery.json']:
        shutil.copyfile(verified/name,evidence/name)
    shutil.copyfile(manifest_path,evidence/'artifact_manifest.json');shutil.copyfile(cuda_receipt,evidence/'CUDA_recovery.json')
    shutil.copyfile(verified.parent/'download_receipt.json',evidence/'download_receipt.json')
    shutil.copyfile(generation_receipt,evidence/'generation_reproduction.json')
    shutil.copyfile(source_run/'metadata.json',evidence/'run_metadata.json')
    shutil.copyfile(source_run/'dataset_manifest.json',evidence/'dataset_manifest.json')
    for name in ['summary.json','validation.csv','generation.json','generation.md']:
        shutil.copyfile(verified/'analysis'/name,out/name)
    shutil.copyfile(verified/'step5000_generation.json',out/'step5000_generation.json')
    native=out/'figures';native.mkdir(exist_ok=True)
    for path in (verified/'analysis/figures').iterdir():shutil.copyfile(path,native/path.name)
    diagnostics=out/'generation_diagnostics.json'
    describe(root,verified/'analysis/generation.json',root/'data/processed/tiny-shakespeare-v3',diagnostics)
    quality=load(diagnostics);grads=np.array([r['gradient_norm'] for r in history])
    best=summary['best_validation'];final=summary['final_validation'];five=next(r for r in validations if r['step']==5000)['validation']
    best_row=history[best['step']-1]
    references={eid:load(root/f'report/pilots/{eid}_seed42/summary.json') for eid in [f'E{i:02d}' for i in range(1,15)]}
    records=[]
    for eid,s in [*references.items(),('F01',summary)]:
        variant=load(root/f'experiments/configs/{eid}.json');config,_,_=load_experiment(root,eid,1024 if variant['tokenizer']=='bpe' else 65)
        records.append({'experiment':eid,'seed':42,'replicates':1,'standard_deviation':None,**config.to_dict(),
                        'head_dimension':config.d_model//config.n_heads,'MLP_hidden_dimension':config.d_model*config.mlp_expansion,
                        'microbatch':4,'accumulation':4096//(4*config.context_length),
                        'sampled_windows_per_update':4096//config.context_length,'target_positions_per_update':4096,
                        'tokenizer':variant['tokenizer'],'parameters':s['parameters'],'updates':s['completed_optimizer_updates'],
                        'best_step':s['best_validation']['step'],'best_BPB':s['best_validation']['bpb'],
                        'final_BPB':s['final_validation']['bpb'],'training_seconds':s['training_seconds'],
                        'validation_seconds':s['validation_seconds'],'peak_GPU_MiB':s['peak_gpu_bytes']/2**20,
                        'training_source_bytes':s['training_source_bytes'],'unique_target_bytes':s['unique_target_bytes'],
                        'source_commit':s['source_commit'],'aggregate_code_hash':s.get('code_hash'),
                        'configuration_hash':s.get('config_hash'),
                        'updates_per_training_second':s['completed_optimizer_updates']/s['training_seconds'],
                        'target_positions_per_training_second':s['training_tokens']/s['training_seconds']})
    atomic_json(out/'master_comparison.json',{'runs':records,'scope':'Fourteen different seed42 configurations plus one combined F01, not independent replications; F01 budget doubled; no pooled SD'})
    with (out/'master_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    comparisons=[]
    for eid in ['E01','E02','E03','E09','E14']:
        baseline=references[eid]['best_validation']['bpb']
        comparisons.append({'reference':eid,'reference_updates':5000,'reference_best_BPB':baseline,
                            'F01_step5000_BPB':five['bpb'],'equal_step_absolute_BPB_difference':five['bpb']-baseline,
                            'F01_best_BPB':best['bpb'],'final_budget_absolute_BPB_difference':best['bpb']-baseline,
                            'final_budget_relative_BPB_percent':100*(best['bpb']/baseline-1)})
    atomic_json(out/'strongest_pilot_comparison.json',{'comparisons':comparisons,'equal_step_is_still_combined_architecture':True,'final_budget_is_10000_vs_5000':True})
    presentation=out/'presentation_support';presentation.mkdir(exist_ok=True);architecture(presentation)
    figures=out/'comparison_figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(records))
    ax.bar(x-.18,[r['best_BPB'] for r in records],width=.36,label='Validation-selected best')
    ax.bar(x+.18,[r['final_BPB'] for r in records],width=.36,label='Final authorized update')
    ax.set_xticks(x,[r['experiment'] for r in records])
    ax.set(ylabel='Conditional validation BPB',title='All 14 seed-42 pilots and F01: observed best versus final')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,'One seed per distinct architecture; no uncertainty estimate. F01 10k/40 validations; pilots 5k/20.',ha='center',fontsize=9)
    savefig(fig,figures,'all15_best_final_BPB')
    fig,ax=plt.subplots(figsize=(9,5))
    for eid in ['E01','E02','E03','E09','E14']:
        rows=list(csv.DictReader((root/f'report/pilots/{eid}_seed42/validation.csv').open(encoding='utf-8')))
        ax.plot([int(r['step']) for r in rows],[float(r['bpb']) for r in rows],marker='.',label=eid,alpha=.8)
    ax.plot([r['step'] for r in validations],[r['validation']['bpb'] for r in validations],color='black',lw=2,marker='.',label='F01')
    ax.set(xlabel='Completed optimizer updates',ylabel='Conditional bits per original source byte',title='Genuine full sliding measurements: combined F01 and strongest pilots')
    ax.legend();ax.grid(alpha=.2);savefig(fig,figures,'validation_BPB_matched_updates')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));chosen=[r for r in records if r['experiment'] in ['E01','E02','E03','E09','E14','F01']]
    for row in chosen:
        axes[0].scatter(row['parameters']/1e6,row['best_BPB']);axes[0].annotate(row['experiment'],(row['parameters']/1e6,row['best_BPB']),xytext=(4,5),textcoords='offset points')
        axes[1].scatter(row['training_seconds']/60,row['best_BPB']);axes[1].annotate(row['experiment'],(row['training_seconds']/60,row['best_BPB']),xytext=(4,5),textcoords='offset points')
    axes[0].set(xlabel='Trainable parameters (millions)',ylabel='Best validation BPB',title='Capacity and quality; exploratory')
    axes[1].set(xlabel='Full authorized training cost (minutes)',ylabel='Best validation BPB',title='F01 10k; pilots 5k')
    for ax in axes:ax.grid(alpha=.2)
    savefig(fig,figures,'capacity_and_training_cost')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));labels=[r['experiment'] for r in chosen]
    axes[0].bar(labels,[r['peak_GPU_MiB'] for r in chosen],color='#34536d');axes[0].set(ylabel='Peak allocated GPU MiB',title='Canonical telemetry scope')
    axes[1].bar(labels,[r['validation_seconds']/60 for r in chosen],color='#b95436');axes[1].set(ylabel='Full-validation minutes',title='F01 40 evaluations; pilots 20')
    savefig(fig,figures,'resource_cost')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot([r['train_seconds']/60 for r in validations],[r['validation']['bpb'] for r in validations],marker='.')
    ax.set(xlabel='Measured accumulated training minutes',ylabel='Validation BPB',title='F01 trajectory versus actual training cost');ax.grid(alpha=.2)
    savefig(fig,figures,'F01_BPB_training_time')
    fig,ax=plt.subplots(figsize=(8,4.5));bars=[r for r in quality['groups'] if r['checkpoint_policy']=='best' and r['greedy']]
    ax.bar([r['experiment'] for r in bars],[r['mean_repeated_8gram_fraction'] for r in bars],color='#34536d')
    ax.set(ylabel='Mean repeated 8-character-gram fraction',title='Five fixed greedy prompts; unequal BPE/character continuation lengths')
    savefig(fig,figures,'fixed_greedy_repetition')
    for path in [native/'training_loss.png',native/'validation_bpb.png',figures/'validation_BPB_matched_updates.png',figures/'resource_cost.png']:
        shutil.copyfile(path,presentation/path.name)
    val_table='\n'.join(f"| {r['step']} | {r['validation']['loss']:.8f} | {r['validation']['bpb']:.8f} | {r['validation']['perplexity']:.6f} |" for r in validations)
    master='\n'.join(f"| {r['experiment']} | {r['d_model']} | {r['n_layers']} | {r['tokenizer']} | {r['parameters']:,} | {r['best_BPB']:.8f} | {r['final_BPB']:.8f} | {r['best_step']} | {r['training_seconds']:.3f} |" for r in records)
    comparison_table='\n'.join(f"| {r['reference']} | {r['reference_best_BPB']:.8f} | {r['F01_step5000_BPB']:.8f} | {r['equal_step_absolute_BPB_difference']:+.8f} | {r['final_budget_absolute_BPB_difference']:+.8f} | {r['final_budget_relative_BPB_percent']:+.3f}% |" for r in comparisons)
    io=load(verified/'io_timing.json');wall=load(verified/'wall_time.json')
    timers={name:sum(r['seconds'] for r in io['operations'] if r['operation']==name) for name in ['checkpoint','export']}
    retrospective=[r for r in validations if r['validation']['bpb']<references['E02']['best_validation']['bpb']]
    slope=float(np.polyfit([r['step'] for r in validations if r['step']>=6000],[r['validation']['bpb'] for r in validations if r['step']>=6000],1)[0]*1000)
    generation=load(verified/'analysis/generation.json');samples=generation['samples']
    quotes=[]
    for greedy,temp,seed in [(True,1.,42),(False,.7,42)]:
        for prompt in generation['protocol']['prompts']:
            s=next(s for s in samples if s['checkpoint_policy']=='best' and s['prompt']==prompt and s['greedy']==greedy and s['temperature']==temp and s['seed']==seed)
            fence='`'*max(3,max([len(x) for x in __import__('re').findall(r'`+',s['text'])]+[0])+1)
            quotes.append(f"### Prompt {prompt!r}; greedy={greedy}, temperature={temp}, seed={seed}\n\nCheckpoint `{s['checkpoint']}`; 256 new BPE tokens, text unchanged.\n\n{fence}text\n{s['text']}\n{fence}\n")
    automatic=[r for r in quality['groups'] if r['experiment']=='F01'];quality_table='\n'.join(f"| {r['checkpoint_policy']} | {r['greedy']} | {r['temperature']} | {r['outputs']} | {r['mean_continuation_characters']:.2f} | {r['mean_repeated_8gram_fraction']:.6f} | {r['replacement_characters']} |" for r in automatic)
    findings={'best':best,'final':final,'step5000':five,'test':test,'gradient_norm':{'median':float(np.median(grads)),
              'p95':float(np.quantile(grads,.95)),'maximum':float(grads.max()),'clipped_updates':int((grads>1).sum()),
              'clip_fraction':float((grads>1).mean())},'late_BPB_slope_per1000_updates_6000_to10000':slope,
              'first_recorded_below_E02_best':{'step':retrospective[0]['step'],'BPB':retrospective[0]['validation']['bpb'],'train_seconds':retrospective[0]['train_seconds']} if retrospective else None,
              'timing':{'training_seconds':summary['training_seconds'],'validation_seconds':summary['validation_seconds'],
                        'checkpoint_seconds':timers['checkpoint'],'export_seconds':timers['export'],'journal_seconds':io['journal_seconds'],
                        'wrapper_seconds_before_archive':wall['pipeline_seconds_before_archive'],
                        'archive_and_CRC_seconds':manifest['archive_and_CRC_seconds'],
                        'measured_sequential_wrapper_component_sum':wall['pipeline_seconds_before_archive']+manifest['archive_and_CRC_seconds'],
                        'independent_download_seconds':transfer['seconds'],
                        'independent_archive_CPU_inspection_seconds':cpu['seconds']},
              'source_byte_sampling':cpu['sampler'],'CPU_recovery':cpu['status'],'CUDA_recovery':cuda['status']}
    findings['selected_best_checkpoint_exposure']={'optimizer_step':best['step'],'predicted_tokens':best_row['training_tokens'],
        'original_source_byte_exposure':best_row['training_bytes'],'training_seconds':best_row['train_seconds']}
    findings['fixed_generation_reproduction']={'slots_exactly_reproduced':100,
        'additional_optimizer_updates':0,'strict_UTF8_invalid_outputs':reproduced['strict_UTF8_invalid_outputs'],
        'receipt':'evidence/research/generation_reproduction.json'}
    atomic_json(out/'findings.json',findings)
    report=f'''# F01 — NanoLM-Final: 10,000-update research analysis

## Acceptance and purpose

One fresh seed42 combined model completed10000 authoritative optimizer updates and40 full sliding validations. Independent archive/CPU and fresh compatible CUDA acceptance receipts passed. Research notebook: https://www.kaggle.com/code/{kernel}. No further architecture, seed or extension was launched. The original three-seed study was discontinued by explicit user decision, not completed as a replicated study.

Primary predictive target: beat E02 best conditional BPB2.02659521. **Observation:** F01 bestBPB={best['bpb']:.8f} at update{best['step']}; finalBPB={final['bpb']:.8f}. Target met: {best['bpb']<references['E02']['best_validation']['bpb']}. Generation quality is assessed separately below and in interpretation.md; lower BPB is not a semantic-coherence result.

## Frozen rationale and confounders

[Pre-training architecture decision](F01_architecture_decision.md) records all14 pilot outcomes and fixed choices. BPE, RoPE, eight blocks and width240 were jointly approved. Context128, six heads, GELU and MLP4x were retained. Hypothesis: the combined capacity/representation might improve validation coding and text; interactions and longer training might instead worsen either outcome. The pilot gains were never added to forecast F01. This is an exploratory combined architecture, not a single-factor ablation.

## Architecture and parameter derivation

```json
{json.dumps(MODEL,indent=2)}
```

```json
{json.dumps(parameter_breakdown(),indent=2)}
```

The eight blocks are independently allocated, pre-LayerNorm, with 240→960→240 GELU MLPs. Six heads use40 coordinates; explicit QKᵀ scores divide by√40, then causal mask/softmax/value aggregation. RoPE rotates adjacent projected Q/K pairs at local positions0–127 using m·10000^(−2i/40); V remains unrotated. No additive sinusoidal or learned positional table is active. Embedding and LM projection are untied. The manual-attention/rotary ideas are related to [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [RoFormer](https://arxiv.org/abs/2104.09864); our basic-layer decoder and its exact implementation are in the frozen project package, without pretrained weights or ready-made Transformer blocks.

![Actual architecture](presentation_support/F01_architecture.png)

## Source, environment and verification

Trained source `{summary['source_commit']}`; original reviewed scientific source `{provenance['frozen_pre_F01_commit']}`. Original aggregate `{provenance['audited_pre_F01_aggregate_hash']}`. Scientific implementation fingerprint `{provenance['scientific_core_fingerprint']}`. F01 aggregate `{summary['code_hash']}`; resolved configuration hash `{summary['config_hash']}`. Adding F01.json changed the aggregate while scientific implementation and all historical configurations stayed unchanged. The independently committed launcher records its own builder commit in notebook metadata; it invokes the frozen execution source.

Actual GPU {environment['gpu']}, CUDA0, Python{environment['python']}, PyTorch{environment['torch']}, runtimeCUDA{environment['cuda']}. Exact package freeze and driver/device memory are in evidence/research/environment.json and hardware.json. Driver capability and torch CUDA runtime are different recorded values. No native CPU wheel replacement or environment upgrade occurred.

CPU combined numerical audit passed; corrected full CPU suite77passed/one CUDA-onlyskip. Actual qualified CUDA suite78passed in39.52s. GitHub CI for source06291e1 and launcher9a93b7a succeeded. Tests include independent head40 rotation/attention, wrong scaling/rotation references, preLN/GELU, shifted manual CE, future invariance, finite block gradients, dropout, deterministic fresh initialization, context cropping, tokenizer roundtrip and same-byte BPB. The initial exactly14-config-files assertion failed after authorized F01 addition and was corrected while retaining all original override assertions. No scientific defect was found.

Mandatory development publication and recovery preceded research. Published134170219-byte smoke archive SHA256fed6afe393ed333e9549d5160e3b0d528ec294aedfc298bad5885794e9c79479 independently verified44files and CPU inference. Fresh CUDA preserved all six state fields and smoke-only resumed/reference tensors, losses and exposure matched. Roundtrip added0updates; continuity executed1resumed plus2reference smoke updates, separate from research. Development benchmark10updates excludes first2for throughput; estimates remain in evidence/development/feasibility.json, not research outcomes.

## Dataset, training and evaluation

Pinned Tiny Shakespeare rawSHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed; canonical BPE checksum2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55. Original train/validation/test bytes1003854/55770/55770, contiguous partitions, no normalization, no cross-boundary windows, train-only1024vocabulary. The complete unchanged manifest is preserved in the native artifacts.

Fresh seed42 initialization; AdamW3e-4constant, weight decay.01, clipping1.0, float32, microbatch4, accumulation8. Thirty-two independent windows/update predict4096BPE targets. Total target exposure40960000; actual original target-byte exposure{summary['training_source_bytes']:,}, unique covered bytes{summary['unique_target_bytes']:,}. Sampler replay confirms private RNG, source-byte sums and union. Input/evaluation byte-context statistics are in findings.json;128BPEtokens do not equal128original bytes.

Full sliding validation every250updates excludes the common256conditioning bytes and scores55514original bytes/24386BPE targets exactly once, batch32, dropout disabled, at most128preceding tokens. Prefix/suffix tokenization separately prevents boundary merges. Local positions reset with truncation. BPB=NLL/(ln2×55514); token loss/perplexity are within-tokenizer diagnostics, not cross-tokenizer rankings.

## All40 observed validation measurements

| Update | Token loss (nats) | Conditional BPB | Token perplexity |
|---:|---:|---:|---:|
{val_table}

Initial recorded training loss {summary['initial_recorded_training']['loss']:.8f} atstep1; initial validation atstep250, notstep0. Final batch loss {summary['final_training_loss']:.8f}; trailing100-update mean {summary['final_training_trailing_100_loss']:.8f}. Best loss {best['loss']:.8f}, BPB {best['bpb']:.8f}, perplexity {best['perplexity']:.6f} atstep{best['step']}. Final loss {final['loss']:.8f}, BPB {final['bpb']:.8f}, perplexity {final['perplexity']:.6f} atstep10000. Full per-step history is external; summary.json and validation.csv preserve actual points.

![Measured loss](figures/combined_loss.png)
![Measured BPB](figures/validation_bpb.png)

## Internal5k-versus10k diagnostic and pilot comparisons

F01 step5000BPB {five['bpb']:.8f}, step10000BPB {final['bpb']:.8f}; absolute change {final['bpb']-five['bpb']:+.8f}. Best uses all40observations. The complete predeclared50-slot step5000grid is in step5000_generation.json. It is an internal diagnostic, not an E02 model. The descriptive6000–10000BPB slope is {slope:+.8f}/1000updates; final minus bestBPB {final['bpb']-best['bpb']:+.8f}. These descriptors do not establish significant overfitting or a mechanistic double-descent claim.

| Reference | Reference5k best BPB | F01at5k BPB | Equal-step difference | F01best minus reference | Relative difference |
|---|---:|---:|---:|---:|---:|
{comparison_table}

Equal-step comparisons still vary multiple architectural factors, parameters, original-byte context/exposure, initialization and compute. Final F01 comparisons additionally double the update budget. One seed supplies no variance or significance estimate. Validation selection across architectures/checkpoints is exploratory; final reserved test assessment follows selection.

![Actual matching validation points](comparison_figures/validation_BPB_matched_updates.png)
![Capacity and full training cost](comparison_figures/capacity_and_training_cost.png)

## Resources and gradient observations

Measured training-only {summary['training_seconds']:.3f}s,40 full validations {summary['validation_seconds']:.3f}s; throughput {10000/summary['training_seconds']:.6f}updates/s and {40960000/summary['training_seconds']:.3f}targets/s. Peak allocated telemetry {summary['peak_gpu_bytes']/2**20:.3f}MiB. Canonical peak is queried before validation after optimizer updates, so earlier validation can contribute; final-validation/postprocessing peak is not separately measured. CUDA memory is allocated memory, not total device/driver memory.

Checkpoint IO {timers['checkpoint']:.3f}s; immutable exports {timers['export']:.3f}s; journal/fsync {io['journal_seconds']:.3f}s. Wrapper through test/generations {wall['pipeline_seconds_before_archive']:.3f}s; archive construction/hash/CRC {manifest['archive_and_CRC_seconds']:.3f}s. These sequential measured components sum {wall['pipeline_seconds_before_archive']+manifest['archive_and_CRC_seconds']:.3f}s; tiny manifest writes, environment bootstrap, queue/publication and network retrieval are separately scoped or unmeasured, not invented as exact whole-session time. Test evaluation {test['seconds']:.3f}s. Native100-slot generation time {generation['seconds']:.3f}s; step5000-generation timing is not separately instrumented.

Global preclip gradient median {np.median(grads):.6f}, p95 {np.quantile(grads,.95):.6f}, maximum {grads.max():.6f}; {int((grads>1).sum())}/10000updates exceeded1.0. Connectivity probes cover every block. Aggregate norms do not establish longitudinal vanishing/exploding layer gradients.

![Resource measurements](comparison_figures/resource_cost.png)

## Complete genuine generation evidence

generation.json/md contains all100best/final fixed slots unchanged. Every setting generates256newBPEtokens; continuation character lengths vary and exceed character-model lengths in many cases. All five prompts, greedy and temperatures.7/1/1.2with seeds42/123/456 remain fixed. Ten predetermined best outputs (all fivegreedy and all fiveattemperature.7/seed42) follow; favorable and unfavorable strings are retained.

{chr(10).join(quotes)}
## Automatic text descriptors and interpretation boundary

| Policy | Greedy | Temperature | Outputs | Mean continuation characters | Repeated8gram fraction | Replacement characters |
|---|---|---:|---:|---:|---:|---:|
{quality_table}

All600matched outputs for F01/E01/E02/E03/E09/E14 are compared in generation_diagnostics.json. Word spans, training-lexicon novelty, speaker labels, distinct ngrams and repeated24/48character phrases are string statistics. Novel words include legitimate names/unseen vocabulary; they are not an automatic grammar score. Prompt-prefix preservation is not semantic prompt adherence. Replacement counts retain native decoding failures without repaired output. Fresh compatibleCUDA independently reproduced every100fixed text outputs exactly and retained their generated token IDs; strict byte reconstruction found{reproduced['strict_UTF8_invalid_outputs']}invalid-UTF8outputs. Original and reproduced decoded texts remain unchanged. Unequal BPE/character lengths affect repetition measures. Manual observations and demonstration-temperature recommendation are in interpretation.md; no unvalidated repetition penalty/top-k/top-p/decoding change was introduced.

![All fixed greedy prompts](comparison_figures/fixed_greedy_repetition.png)

## Frozen final test assessment

Selection receipt fixes checkpoint SHA256 `{freeze['checkpoint_sha256']}` atstep{freeze['best_step']} using validation BPB only. The source writes this immutable freeze before invoking one full sliding test evaluation. Test loss {test['metrics']['loss']:.8f}, BPB {test['metrics']['bpb']:.8f}, perplexity {test['metrics']['perplexity']:.6f}, scoredbytes{test['metrics']['scored_bytes']}, scoredtokens{test['metrics']['scored_tokens']}, prefix256, context128BPEtokens. No weights/checkpoint/architecture/decoding choices changed from test results, and no retraining was launched. The conditioning-constrained metric is not unconditional corpus entropy.

## Publication and independent recovery

Saved research output version1 at https://www.kaggle.com/code/{kernel}. Archive {manifest['archive_bytes']:,}bytes; SHA256 `{manifest['archive_sha256']}`. Exact{cpu['verified_file_count']}file inventory, CRC, safe paths, every file hash,40optimizer/scheduler counters,10000history rows,40full validations and41snapshot manifests independently checked. Fresh qualifiedCUDA verified all native snapshot files and complete model/optimizer/scheduler/scaler/RNG/sampling/history roundtrip10000→10000with0additional updates. Best/finalCPU and CUDA inference receipts identify each checksum. Raw snapshots remain in published outputs; ZIP contains every research checkpoint once plus all native analysis/evidence and snapshot index.

Independent Windows archive transfer took {transfer['seconds']:.3f}s, separately from training and notebook execution. Completed ranges and failed-attempt partial transfer evidence remain on D:; retries do not represent optimizer replay. Local complete archive/CPU inspection took {cpu['seconds']:.3f}s. The actual transfer duration must not be substituted for training-only cost or concealed inside an unmeasured whole-platform wall-clock claim.

Authoritative10000updates; recorded journal attempts{cpu['journal_recorded_update_attempts']}; uninterrupted sequential journal={cpu['uninterrupted_sequential_journal']}. If any resume/failure/attempt discrepancy exists, the receipt records it and physical work needs explicit reconciliation; no attempt history is silently merged. HTTP chunked log-stream reconnects are preserved separately from training evidence and do not count as optimizer restarts. The initial development launcher redirected child logs tobootstrap.log; later committed launcher exposes progress and records its own commit without changing frozen scientific execution.

## Limitations and final interpretation

The predictive outcome, actual generations and resource costs are separate results. A combined six-million-parameter model and doubled budget cannot isolate tokenizer/RoPE/depth/width contributions. Matchingseed42 does not imply identical initialization tensors across changed shapes. Original-byte exposure, effective byte context, single contiguous corpus, repeated validation-based choices, one seed and separate GPU sessions restrict generalization. No statistical-significance claim, general model superiority, reasoning competence or global memorization rate follows from these observations. Literal training-passage reproduction is recorded separately in fixed20_training_overlaps.json. See interpretation.md for evidence-backed qualitative findings and demonstration recommendation. Further training/decoding work requires new authorization; this task stops at10000.
'''
    (out/'F01_10000_analysis.md').write_text(readable_prose(report),encoding='utf-8')
    (out/'E01_E14_consolidated_analysis.md').write_text('# All fourteen seed42 pilots and accepted F01\n\n| Experiment | Width | Blocks | Tokenizer | Parameters | Best BPB | Final BPB | Best step | Training seconds |\n|---|---:|---:|---|---:|---:|---:|---:|---:|\n'+master+'\n\n'+
        'The 14 configurations are different architectures, not 14 replications. F01 adds combined factors and 10,000 updates; others use 5,000. Mean/SD across seeds cannot be estimated. All original configs/reports remain unchanged. See master_comparison.csv/JSON for complete shapes, source-byte exposure and resource fields, F01_architecture_decision.md for the frozen rationale, and F01_10000_analysis.md for measured comparisons/negative outcomes. E02 BPE and F01 are ranked using same-byte BPB; token perplexities are incomparable across tokenizers. Context/depth/width affect information/capacity/compute together. E07 replay accounting remains in its original report; retained trajectory time is not substituted for total known physical work. E12 records no observed replay. No test-set model selection occurred.\n\n![Actual validation curves](comparison_figures/validation_BPB_matched_updates.png)\n',encoding='utf-8')
    (presentation/'README.md').write_text('# F01 presentation evidence\n\nArchitecture and RoPE diagram are mathematical/implementation illustrations; loss/BPB/resource charts contain genuine recorded data. Full findings, limitations and unedited generated texts are linked in ../F01_10000_analysis.md and ../generation.md. Single seed; combined architecture and doubled budget; no inference that lower BPB proves coherent text. No additional training was launched.\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','verified','manifest','cuda-receipt','generation-receipt','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kernel',required=True);a=p.parse_args();write(a.root,a.verified,a.manifest,a.cuda_receipt,a.generation_receipt,a.output,a.kernel)
