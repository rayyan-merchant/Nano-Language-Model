# Assignment evidence index

All ten Section 12 items are provided, despite Section 4's earlier “at least 6”
wording. New educational diagrams are separate from immutable research results.
The [audit receipt](../verification/assignment_audit.json) records source hashes,
actual constructed counts, tensor traces and numerical checks. All PNGs were
visually inspected; reused figures remain at their historical paths.

| Figure | Handout requirement | Evidence | Model/source | Verified status and report-ready caption | Caveat |
|---|---|---|---|---|---|
| 1 | Three-token hand-calculated self-attention | [Original PDF](../../handwritten-assignment.pdf), [corrected companion](hand_calculation_audit.md), [figure](figure_01_attention_derivation.png) | Educational Q=K=V, d_k=d_v=2 | PASS with disclosed rounding correction: “Three-token scaled attention, causal score masking, row normalization and value aggregation.” | New companion is typed, not retroactively handwritten; original remains unchanged |
| 2 | Causal-mask visualization | [Mask and probabilities](figure_02_causal_mask.png) | Actual `scaled_attention` convention | PASS: “Boolean future exclusion sets upper-triangular scores to −∞ before softmax, giving zero future probability.” | Values are the educational example, not trained attention |
| 3 | Multi-head tensor shapes | [Trace](figure_03_tensor_shapes.png), [machine receipt](../verification/assignment_audit.json) | Constructed E02 config, batch 1, length 12 | PASS: “Actual projections split width 192 into six 32-dimensional heads, aggregate values and merge before output projection.” | Fresh structural trace; Figure 7 separately uses trained weights |
| 4 | Sinusoidal positional encoding | [Heatmap and frequency slices](figure_04_sinusoidal.png) | Actual `sinusoidal(128,192)` | PASS: “Sine/cosine position features vary at dimension-dependent frequencies and add to token embeddings in E02.” | RoPE is explained additionally, not substituted for the required sinusoidal encoding |
| 5 | Architecture and parameter counts | [Pipeline schematic/counts](figure_05_architecture.png) | Actual E02/B01/F01 construction | PASS: “Pre-LN causal decoder architecture; selected models contain 2,173,056 / 3,062,784 / 6,046,560 parameters.” | E02 adds sinusoidal positions; B01/F01 rotate Q/K with RoPE; counts include embeddings and untied head |
| 6 | Training versus validation loss | [Existing F01 figure](../../../report/final/figures/combined_loss.png), [validation CSV](../../../report/final/validation.csv), [summary](../../../report/final/summary.json) | Genuine F01, 10,000 updates | PASS: “Training NLL continues falling while held-out full-sliding NLL deteriorates after the selected 1,750-update checkpoint.” | Training is a 100-step trailing mean; validation observations are every 250 steps, joined for readability, not additional measurements; no step-zero validation |
| 7 | Trained attention heatmap | [Existing PNG](../verification/colab-final/receipts/attention/trained_attention.png), [receipt](../verification/colab-final/receipts/attention/receipt.json) | B01 selected step 3000, layer 0/head 0, 10 BPE tokens | PASS: “Trained causal attention over the fixed Romeo prompt has normalized rows and exactly zero future probability.” | One head/prompt is an example, not an explanation of all model behavior; coordinates are query rows and key columns |
| 8 | At least five genuine text examples | [Full unedited text](figure_08_generations.md), [display figure](figure_08_generations.png), [complete 30-case receipt](../verification/colab-final/receipts/generation/receipt.json) | All five B01 greedy prompts, step 3000 | PASS: “The five predetermined prompts produce dialogue fragments, malformed words and repeated phrases under identical decoding settings.” | Final CPU demo: 32 new tokens, seed 42, greedy, T=.7 ignored by argmax; image display wraps text, Markdown preserves actual line breaks; not the original 256-token research grid |
| 9 | Ablation comparison, at least three factors | [Depth/heads/context figure](figure_09_ablations.png), [all-14 CSV](figure_09_ablations.csv), [full analysis](../../../report/final/E01_E14_consolidated_analysis.md) | Genuine E01–E14 seed 42 | PASS: “Best validation conditional BPB for three controlled-factor families under a fixed predicted-target budget.” | Single seed; capacity, conditioning, sampling and compute confounds; see analysis below |
| 10 | CLI or API inference | [Actual API request/response](figure_10_api.png), [receipt](../verification/colab-final/receipts/api/receipt.json) | B01 step 3000, genuine CPU FastAPI TestClient | PASS: “POST /v1/generate returns actual selected-checkpoint text with request settings and checkpoint metadata.” | Historical executed local API evidence, not proof of present internet uptime; parameter count/context verified from the same model receipt |

## Quantitative interpretation and caveats

F01's selected update 1750 has validation NLL 3.13024499 and BPB 1.98376817;
its final update 10000 has NLL 4.81814590 and BPB 3.05346210. Its final raw training
loss is .92398473 and trailing-100 mean .94353615. Thus the selected inference
package intentionally uses the validation-best state rather than the terminal
training state. [B01 selection](../../../report/balanced/B01_model_selection.md)
similarly chooses update 3000 (BPB 1.98493095) from its 4500-update completed run.
These are observed generalization deterioration patterns, not evidence of a
universal double-descent mechanism or independent-seed significance.

Figure 9 uses best validation BPB, not token perplexity. Its companion CSV retains
actual parameters, retained training time, validation cost, memory, original-byte
exposure and provenance for all fourteen experiments. Depth 4/6/8 gives best BPB
2.21405793/2.14548940/2.10014737, with increasing capacity and computation.
Heads 4/6/8 gives 2.20537989/2.21405793/2.21051529 at constant width but changing
head dimensions. Context 64/128/256 gives 2.15396251/2.21405793/2.25627886 with
accumulation 16/8/4 and sampled windows 64/32/16 per update. All predict 4096
positions/update, but equal target counts do not equal conditioning or compute.

E02 BPE has a different vocabulary, byte exposure and effective source-text context.
Its token perplexity cannot be compared directly with character experiments.
F01/B01 combine factors and are not additional single-factor ablations.
Run-time differences reflect hardware/runtime conditions as well as architecture;
E07 retained training time excludes known superseded attempts after interruption
(207 replayed updates), so it is not its full physical training cost. See the
accepted consolidated analysis for detailed replay accounting and qualitative
generation comparisons. Greedy loops, malformed text and byte-decoding failures
are evidence-backed limitations, not concealed unsuccessful examples.

## Additional required conceptual evidence

[MLM versus CLM and all seven concepts](concepts_and_objectives.md) includes an
[educational visibility figure](mlm_vs_clm.png), shifted targets, objective
equations in words, and a numerical masked/unmasked comparison. No BERT model
was trained or used. The Transformer/recurrence comparison is reasoned complexity,
not a fabricated benchmark.

## Rebuild and verify

```text
python -m tools.assignment_evidence
python -m pytest tests/test_assignment_evidence.py -q
```

Run from an installed checkout. This performs CPU inference on fresh structural
models and scalar arithmetic, reads frozen receipts, and generates presentation
artifacts only. It does not train, download weights, evaluate original test data,
or regenerate accepted research figures. Figure 6 and Figure 7 are linked in place.
Current measured verification and unresolved human deliveries are recorded in
[assignment readiness](../assignment_readiness.md).
