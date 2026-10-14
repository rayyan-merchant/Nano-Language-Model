"""Reproducible pre-run decision, historical metrics and fixed comparison grid."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from nano_lm.artifacts import atomic_json, file_hash
from scripts.final.protocol import BASE, provenance, configuration, parameter_breakdown


def write(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    proof=provenance(root);records=[];observations=[]
    for i in range(1,15):
        eid=f'E{i:02d}';folder=root/'report/pilots'/f'{eid}_seed42'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        config=json.loads((root/f'experiments/configs/{eid}.json').read_text())
        report=folder/f'{eid}_seed42_analysis.md';report_text=report.read_text(encoding='utf-8')
        raw=subprocess.check_output(['git','show',f'{BASE}:{report.relative_to(root).as_posix()}'],cwd=root)
        assert raw.decode('utf-8').replace('\r\n','\n')==report_text
        for path in folder.glob('*interpretation*'):
            path.read_text(encoding='utf-8')
        grid=json.loads((folder/'generation.json').read_text(encoding='utf-8'))
        assert len(grid['samples'])==100 and summary['completed_optimizer_updates']==5000
        best=summary['best_validation'];final=summary['final_validation']
        assert best['scored_bytes']==final['scored_bytes']==55514
        records.append({'experiment':eid,'factor':config['factor'],'tokenizer':config['tokenizer'],
                        'model_overrides':config['model_overrides'],'parameters':summary['parameters'],
                        'best_step':best['step'],'best_BPB':best['bpb'],'final_BPB':final['bpb'],
                        'training_seconds':summary['training_seconds'],'validation_seconds':summary['validation_seconds'],
                        'peak_GPU_bytes':summary['peak_gpu_bytes'],'training_source_bytes':summary['training_source_bytes'],
                        'source_commit':summary['source_commit'],'report_sha256':file_hash(report)})
        groups={}
        for s in grid['samples']:
            assert s['text'].startswith(s['prompt'])
            text=s['text'][len(s['prompt']):]
            grams=[text[j:j+8] for j in range(max(0,len(text)-7))]
            repeated=1-len(set(grams))/len(grams) if grams else 0
            key=f"{s['checkpoint_policy']}|{'greedy' if s['greedy'] else s['temperature']}"
            groups.setdefault(key,[]).append({'characters':len(text),'replacement_characters':text.count('\ufffd'),
                                            'repeated_8gram_fraction':repeated})
            if eid in {'E01','E02','E03','E09','E14'}:
                observations.append({'experiment':eid,**s,'continuation_characters':len(text),
                                     'replacement_characters':text.count('\ufffd'),'repeated_8gram_fraction':repeated})
        records[-1]['automatic_generation_descriptors']={k:{'outputs':len(v),
            'mean_characters':sum(x['characters'] for x in v)/len(v),
            'replacement_characters':sum(x['replacement_characters'] for x in v),
            'mean_repeated_8gram_fraction':sum(x['repeated_8gram_fraction'] for x in v)/len(v)} for k,v in groups.items()}
    atomic_json(out/'pilot_master_metrics.json',{'frozen_reviewed_main':BASE,'pilots':records,
                'F01_status':'architecture frozen; untrained; all GPU and persistence gates pending',
                'scientific_provenance':proof,'F01_model':configuration(root).to_dict(),'parameter_breakdown':parameter_breakdown()})
    fields=[k for k in records[0] if k not in {'model_overrides','automatic_generation_descriptors'}]
    with (out/'pilot_master_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    atomic_json(out/'fixed_pilot_generation_comparison.json',{'selection':'All 100 slots per E01/E02/E03/E09/E14, unedited; unequal generated character lengths', 'samples':observations})
    table='\n'.join(f"| {r['experiment']} | {r['factor']} | {r['parameters']:,} | {r['best_BPB']:.8f} | {r['final_BPB']:.8f} | {r['best_step']} |" for r in records)
    rationale='''The approved final model combines BPE, RoPE, eight blocks and width 240. E02 had the lowest recorded BPB; E03 improved character likelihood; E09 was the strongest depth result; E14 improved width-192 likelihood. These are observed single-factor outcomes, not measurements of their interaction. No additive percentage forecast is justified.

Context 128 is retained despite E11's favorable BPB: E11 doubled the sampling windows/update, increased training time and worsened greedy repetition; E12 did not improve baseline likelihood. Six heads remain because E06's gain was small and single-seed. GELU and MLP expansion four remain: ReLU E05 and reduced MLP E10 did not improve baseline BPB. Learned positions E04, width144 E13 and eight heads E07 also did not establish a reason to override the approved combined architecture.

All five prompts and all decoding settings from E01/E02/E03/E09/E14 are retained in fixed_pilot_generation_comparison.json. Repeated-eight-gram and replacement-character statistics are automatic string descriptors, not grammar/coherence scores. BPE emits unequal character lengths at 256 new tokens; this prevents naïve length-free repetition interpretation. Historical report interpretations and complete raw grids remain the primary qualitative evidence. E02's longer dialogue still includes loops/malformed words and replacement characters; RoPE's favorable BPB did not guarantee reduced greedy repetition. E09/E14 reduced measured greedy repetition without establishing semantic competence.

Prediction/hypothesis: greater capacity with BPE and RoPE may improve same-byte validation likelihood; interaction and longer optimization may instead degrade generalization or generation. Only the single approved F01 trajectory can test this exploratory hypothesis. Target BPB below E02 2.02659521 is not guaranteed. Text quality is a separate objective.

The user explicitly discontinued the three-seed grid on 2026-10-14 and separately authorized F01 seed42 for 10000 updates. Historical E01–E14 definitions and evidence remain unchanged. No uncertainty across seeds can be estimated. Architecture, capacity, context in original bytes, tokenizer, source-byte exposure, initialization and compute are jointly changed. F01 at5k is an exploratory equal-step comparison; F01 at10k also changes training budget.

Selection is the lowest canonical validation BPB over40 measurements. For this fixed tokenizer/byte coverage, minimum token loss selects the same checkpoint. Freeze selection identity before one reserved test evaluation; never change it based on test results. The full 100-slot best/final generation grid remains unchanged, and a separate full50-slot step5000 grid is an internal diagnostic. No secondary length-matched inference is planned, avoiding post-hoc sampling choices.
'''
    text=f'''# F01 architecture decision — frozen before training

Reviewed main: `{BASE}`. F01 is authorized but has not trained. No pilot result file was changed.

## Observed fourteen-pilot evidence

| Experiment | Factor | Parameters | Best BPB | Final BPB | Best update |
|---|---|---:|---:|---:|---:|
{table}

## Decision, hypotheses and limitations

{rationale}
## Exact approved model

```json
{json.dumps(configuration(root).to_dict(),indent=2)}
```

## Independent parameter derivation

```json
{json.dumps(parameter_breakdown(),indent=2)}
```

Untied embedding and output matrices each contribute245760; each block694320, eight blocks5554560, final LayerNorm480: total6046560. RoPE contributes no trainable parameters.

## Provenance and preflight gates

Scientific core fingerprint: `{proof['scientific_core_fingerprint']}`. Definition and file hashes are in pilot_master_metrics.json. Original aggregate: `{proof['audited_pre_F01_aggregate_hash']}`. The F01 config intentionally changes the aggregate; record the new value and clean training SHA in execution receipts. All40 validations score55514 bytes after256 conditioning bytes, context128BPEtokens, batch32. Training: freshseed42, AdamW3e-4constant/wd.01/clip1/float32/micro4/acc8/4096targets, exactly10000updates.

Required before research: independent CPU/CUDA numerical checks, full suite, qualified freeze/T4, real benchmark feasibility, durable smoke publication/retrieval, exact freshCUDA full-state restoration and smoke-only continuation. No research launch until all gates pass. Kaggle documents12h GPU sessions (https://www.kaggle.com/docs/notebooks); the wrapper budgets10h plus2h provisioning/publication reserve. This is a conservative envelope, not a guarantee against interruption.
'''
    (out/'F01_architecture_decision.md').write_text(text,encoding='utf-8')
    (out/'E01_E14_consolidated_analysis.md').write_text('# Recorded fourteen-pilot comparison; F01 pending\n\n'+table+'\n\n'+rationale,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();write(a.root,a.output)
