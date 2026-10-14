# F01 architecture decision — frozen before training

Reviewed main: `e4f768d53038662f0f24bcb9cb84a5210ad38a85`. F01 is authorized but has not trained. No pilot result file was changed.

## Observed fourteen-pilot evidence

| Experiment | Factor | Parameters | Best BPB | Final BPB | Best update |
|---|---|---:|---:|---:|---:|
| E01 | baseline | 1,804,800 | 2.21405793 | 2.21599704 | 4750 |
| E02 | tokenizer | 2,173,056 | 2.02659521 | 2.02659521 | 5000 |
| E03 | position | 1,804,800 | 2.07327076 | 2.07327076 | 5000 |
| E04 | position | 1,829,376 | 2.23778869 | 2.23778869 | 5000 |
| E05 | activation | 1,804,800 | 2.24266813 | 2.24266813 | 5000 |
| E06 | heads | 1,804,800 | 2.20537989 | 2.20617553 | 4750 |
| E07 | heads | 1,804,800 | 2.21051529 | 2.21051529 | 5000 |
| E08 | depth | 2,694,528 | 2.14548940 | 2.14758132 | 4750 |
| E09 | depth | 3,584,256 | 2.10014737 | 2.11680859 | 4750 |
| E10 | mlp_expansion | 1,213,440 | 2.26711868 | 2.26711868 | 5000 |
| E11 | context | 1,804,800 | 2.15396251 | 2.15396251 | 5000 |
| E12 | context | 1,804,800 | 2.25627886 | 2.25627886 | 5000 |
| E13 | width | 1,021,824 | 2.32495848 | 2.32495848 | 5000 |
| E14 | width | 2,808,960 | 2.14189208 | 2.14505608 | 4750 |

## Decision, hypotheses and limitations

The approved final model combines BPE, RoPE, eight blocks and width 240. E02 had the lowest recorded BPB; E03 improved character likelihood; E09 was the strongest depth result; E14 improved width-192 likelihood. These are observed single-factor outcomes, not measurements of their interaction. No additive percentage forecast is justified.

Context 128 is retained despite E11's favorable BPB: E11 doubled the sampling windows/update, increased training time and worsened greedy repetition; E12 did not improve baseline likelihood. Six heads remain because E06's gain was small and single-seed. GELU and MLP expansion four remain: ReLU E05 and reduced MLP E10 did not improve baseline BPB. Learned positions E04, width144 E13 and eight heads E07 also did not establish a reason to override the approved combined architecture.

All five prompts and all decoding settings from E01/E02/E03/E09/E14 are retained in fixed_pilot_generation_comparison.json. Repeated-eight-gram and replacement-character statistics are automatic string descriptors, not grammar/coherence scores. BPE emits unequal character lengths at 256 new tokens; this prevents naïve length-free repetition interpretation. Historical report interpretations and complete raw grids remain the primary qualitative evidence. E02's longer dialogue still includes loops/malformed words and replacement characters; RoPE's favorable BPB did not guarantee reduced greedy repetition. E09/E14 reduced measured greedy repetition without establishing semantic competence.

Prediction/hypothesis: greater capacity with BPE and RoPE may improve same-byte validation likelihood; interaction and longer optimization may instead degrade generalization or generation. Only the single approved F01 trajectory can test this exploratory hypothesis. Target BPB below E02 2.02659521 is not guaranteed. Text quality is a separate objective.

The user explicitly discontinued the three-seed grid on 2026-10-14 and separately authorized F01 seed42 for 10000 updates. Historical E01–E14 definitions and evidence remain unchanged. No uncertainty across seeds can be estimated. Architecture, capacity, context in original bytes, tokenizer, source-byte exposure, initialization and compute are jointly changed. F01 at5k is an exploratory equal-step comparison; F01 at10k also changes training budget.

Selection is the lowest canonical validation BPB over40 measurements. For this fixed tokenizer/byte coverage, minimum token loss selects the same checkpoint. Freeze selection identity before one reserved test evaluation; never change it based on test results. The full 100-slot best/final generation grid remains unchanged, and a separate full50-slot step5000 grid is an internal diagnostic. No secondary length-matched inference is planned, avoiding post-hoc sampling choices.

## Exact approved model

```json
{
  "vocab_size": 1024,
  "d_model": 240,
  "n_heads": 6,
  "n_layers": 8,
  "context_length": 128,
  "mlp_expansion": 4,
  "activation": "gelu",
  "position": "rope",
  "dropout": 0.1
}
```

## Independent parameter derivation

```json
{
  "token_embeddings": 245760,
  "per_block": {
    "qkv_projections": 173520,
    "attention_output": 57840,
    "two_pre_LayerNorms": 960,
    "MLP_first": 231360,
    "MLP_second": 230640
  },
  "eight_blocks": 5554560,
  "final_LayerNorm": 480,
  "untied_LM_head": 245760,
  "total": 6046560
}
```

Untied embedding and output matrices each contribute245760; each block694320, eight blocks5554560, final LayerNorm480: total6046560. RoPE contributes no trainable parameters.

## Provenance and preflight gates

Scientific core fingerprint: `d5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8`. Definition and file hashes are in pilot_master_metrics.json. Original aggregate: `cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`. The F01 config intentionally changes the aggregate; record the new value and clean training SHA in execution receipts. All40 validations score55514 bytes after256 conditioning bytes, context128BPEtokens, batch32. Training: freshseed42, AdamW3e-4constant/wd.01/clip1/float32/micro4/acc8/4096targets, exactly10000updates.

Required before research: independent CPU/CUDA numerical checks, full suite, qualified freeze/T4, real benchmark feasibility, durable smoke publication/retrieval, exact freshCUDA full-state restoration and smoke-only continuation. No research launch until all gates pass. Kaggle documents12h GPU sessions (https://www.kaggle.com/docs/notebooks); the wrapper budgets10h plus2h provisioning/publication reserve. This is a conservative envelope, not a guarantee against interruption.
