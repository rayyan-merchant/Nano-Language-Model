# B01 model selection — prediction, compactness and genuine text

**Conclusion:** retain B01's validation-best update 3,000 as a compact predictive
candidate. It nearly matches F01's selected minimum with about half its parameters
and improves on E02's observed minimum. It does **not** establish better generated
text, a global optimum or statistical superiority. E02 remains the smaller,
cheaper measured alternative. No additional training is recommended without a
separately approved question and evaluation plan.

This interpretation uses the original fixed outputs and verified measurements;
all comparative tables/figures are reproducible from `summary.json`,
`matched_validation_comparison.csv` and `generation_comparison.json`. No original
test likelihood or external assessment was performed for B01.

## Hypothesis and observation boundaries

The pre-run hypothesis was that reducing F01's combined capacity while preserving
BPE/RoPE and using frozen validation stopping might retain competitive prediction
and avoid excessive late training. This was a separately approved combined model,
not an additional single-factor ablation. Width, depth, head/MLP dimensions,
initialization and stopping opportunities differ across reference models.

The hypothesis did not imply that small size fixes grammar, that clipping caused
F01's deterioration, or that combining individual pilot winners adds their gains.
Actual data support a narrower conclusion about one seed and one contiguous
Tiny Shakespeare validation suffix.

## Predictive quality: selected minima and final states

| Model | Parameters | Actual updates | Selected best update | Best BPB | Final BPB |
|---|---:|---:|---:|---:|---:|
| E02 | 2,173,056 | 5,000 | 5,000 | 2.02659521 | 2.02659521 |
| F01 | 6,046,560 | 10,000 | 1,750 | 1.98376817 | 3.05346210 |
| B01 | 3,062,784 | 4,500 | 3,000 | 1.98493095 | 2.04445796 |

**Observation:** B01 best-minus-E02 best is −0.04166426 BPB, a 2.0559% decrease.
B01 best-minus-F01 best is +0.00116279 BPB, approximately 0.0586% higher. Both use
the same canonical BPE and context 128 with the same 55,514 scored source bytes.
This supports a directly aligned conditional predictive comparison; it supplies
no uncertainty estimate or evidence that the tiny F01/B01 gap is reproducible.

**Observation:** B01 final-minus-best is +0.05952701 BPB, about 3.00% worse. Final
B01 is also worse than E02's selected/final BPB. Its training loss nevertheless
continues declining: trailing means at 3k/4k are 2.6929/2.5042, and the final
trailing-100 mean is 2.4229. This growing generalization gap and sustained late
validation increase are consistent with overfitting on this run. They do not
identify a causal mechanism or prove every later checkpoint would deteriorate.

**Observation:** all six post-minimum validations exceed the 3,000 minimum,
despite a small downward fluctuation at 3,750 relative to 3,500. Patience increases
at 3,250/3,500/3,750/4,000/4,250/4,500 and stops at six. The small improvement at
2,500 did not reset the 0.001 threshold; the cumulative gain at 2,750 did. The
absolute best was always tracked independently from that patience reference.

**Interpretation:** predefined stopping saved the unexecuted last 500 authorized
updates and retained the earlier best model; it did not prevent all deterioration
or stop exactly at the unknown optimum. No B01 4,750/5,000/10,000 likelihood is
invented. The relevant F01 comparator is its early selected minimum, not its poor
final state. The counterfactual B01 curve under another stop rule was not measured.

![Measured conditional trajectories](https://raw.githubusercontent.com/rayyan-merchant/Nano-Language-Model/8f88edee89dfceb6d0a8c84e385a886c4e058356/report/balanced/comparison_figures/validation_BPB_comparison.png)

## Compactness, cost and actual inference

**Observation:** B01 removes 2,983,776 parameters relative to F01 (49.3467%) but
adds 889,728 relative to E02 (40.9436%). Its measured best/final full resumable
checkpoints are 38,406,090 / 38,640,522 bytes; these include optimizer and history,
not just parameters. Additional capacity over E02 is justified only by the
observed selected likelihood improvement, not by a blanket quality claim.

| Model | Whole trajectory training | Full validations | Saved peak allocated MiB |
|---|---:|---:|---:|
| E02 | 1,128.893 s | 198.577 s | 157.330 |
| F01 | 4,269.206 s | 945.441 s | 253.975 |
| B01 | 1,827.392 s | 264.547 s | 175.943 |

These are different budgets and sessions, not an equal-compute comparison.
B01 averages 2.4625 completed updates/s and about 10,086.51 BPE target tokens/s.
Reaching its selected checkpoint cost 1,239.460 training seconds and 12,288,000
predicted positions/29,994,269 original target bytes. F01 reached its early selected
checkpoint in 747.174 training seconds; E02's selected 5k state cost 1,128.893 s.
Thus B01 did not achieve a cheaper observed selected minimum merely by being
smaller than F01. No matched-session inference speedup against either is claimed.

B01's allocator peak is sampled training telemetry and resets per planned
segment. Validation/inference peaks were not separately recorded; F01's continuous
telemetry can include prior validation. Driver-wide device usage differs again.
The saved peaks are preserved, but percentage memory savings cannot establish
an equivalently measured end-to-end resource advantage.

On the genuine fixed grid, best/final first requests took 3.9527 / 3.7487 s;
the remaining 49 per policy averaged 3.8527 / 3.9048 s for 256 new tokens,
or 66.447 / 65.560 generated BPE tokens/s. CUDA synchronization brackets original
generation; timing excludes tokenization, decoding, model load and CUDA transfer.
The canonical generator recomputes the cropped context; this is not a cached
serving benchmark. Actual CPU deserialize/build times were 0.2017 / 0.0824 s,
with filesystem cache effects. No equivalent E02/F01 request timings are available.

## Optimization, repeated exposure and hypotheses

B01's preclip norm median/p95/max is 0.72513 / 0.95494 / 1.04393; only 15 of
4,500 updates exceeded 1.0 (0.3333%). Per-250 interval measurements remain in
`gradient_intervals.json`. F01's full 10k record exceeded the threshold on 6,988
updates (69.88%). Different model shapes, optimization stages and horizons
preclude inferring that clipping causes overfitting or that B01 avoids it.

B01 executed 144,000 sampled windows, 18,432,000 predicted BPE positions and
44,992,723 original target-byte exposures. Independent RNG/offset union replay
finds 1,003,827 unique target bytes out of the 1,003,854 training-byte partition;
this is not a claim of complete epochs or universal target coverage. Equal BPE
token budgets do not imply equal source-byte exposures, FLOPs or wall time.

Capacity, repeated source exposure, distribution mismatch, constant-LR optimization
and component interactions remain hypotheses. No alternate learning rate,
precision, dropout, tokenizer fit, clipping threshold or extra seed was tried.

## Genuine generation: an important negative result

The complete 100 B01 slots and 300 matched E02/F01/B01 strings are retained.
All five greedy and all 45 sampled outputs from each B01 checkpoint were read.
Detailed zero-based slot observations are in
[B01_generation_observations.md](B01_generation_observations.md).

| Selected best outputs | E02 | F01 | B01 |
|---|---:|---:|---:|
| Greedy mean repeated 8-character-gram fraction (five prompts) | 0.62577 | 0.52467 | 0.64201 |
| Temperature 0.7 mean fraction (15 prompt/seed slots) | 0.05885 | 0.02855 | 0.03854 |

**Observation:** B01's best greedy repetition descriptor is worse than both
references. JULIET repeatedly says “I know not what I have done to do it”; the
citizen prompt devolves into repeated LUCIO labels, and the king prompt loops
on kingship. Its final JULIET output repeats Nurse “O, help!” many times. Lower
BPB therefore did not remove greedy collapse.

**Observation:** B01 final greedy mean fraction falls to 0.38953, while final
temperature0.7 rises to 0.05823. A lower single descriptor does not make the final
checkpoint more grammatical or justify changing the frozen selection policy.
F01's late final state likewise has low greedy repetition yet documented literal
training copying and poor likelihood. Repetition is one failure mode, not a
general quality score.

Temperature0.7 often disrupts loops, but samples still contain malformed words,
abrupt changes between plays' speakers, inconsistent roles and broken grammar.
At1.0/1.2, malformed labels and invented morphology become prominent. Similar
seed-conditioned openers across prompts show limited prompt differentiation;
fixed sampling seeds deliberately couple RNG and are not independent replicates.

Five B01 outputs contain invalid raw UTF8 sequences: best slots5/8/18/48 and
final slot78. They remain unchanged in decoded text and raw256-token receipts.
Best/final weights differ. E02's identical best/final outputs are policy duplicates,
not independent evidence. F01's five invalid best outputs include six replacement
characters, distinguishing output count from decoder-character count.

For a bounded demonstration, use B01's validation-best state with temperature0.7
from the existing grid, showing the limitations and full grid alongside excerpts.
This is not a hidden decoding modification, a guarantee of coherence or an
additional sampling search. No top-k/top-p/repetition penalty was introduced.

## Training-text overlap: observed bounds, not an immunity claim

All100 B01 continuations, excluding prompts, were checked for their longest exact
substring against training text only. The best-policy maximum is34 characters:
`\n\nKING RICHARD III:\nMadam, I have ` in ROMEO/temperature0.7/seed123,
at training offset280,528. The final maximum is40 characters:
`.\n\nHENRY BOLINGBROKE:\nMy gracious lord, ` in king/temperature0.7/seed456,
at offset396,775. These largely contain recurring labels/common phrasing.
No long verbatim passage analogous to F01's final371-character JULIET output was
found in this fixed B01 grid. That absence does not establish absence of
memorization, near-copying, rare-trigger copying or retrieval under other prompts.

Low overlap and diverse gibberish also do not establish useful generalization.
Byte validity, grammar, dialogue continuity and likelihood must be considered
separately; automated descriptors are not substitutes for a blinded evaluation.

## Independent assessment proposal — not executed

The original test result was exposed in F01 and is not reused for B01 selection,
stopping, architecture or decoding. An independent curator could assemble a
held-out set of complete public-domain plays demonstrably absent from the entire
original Tiny Shakespeare source, checking provenance and exact/near-duplicate
overlap before measurement. No named work is assumed absent without verification.

Before any new assessment, freeze E02/F01/B01 checkpoint hashes, identical BPE
artifact, context128, source normalization, conditioning exclusions, boundary-byte
policy, batching and source-byte aggregation. Predeclare what genre/domain shift
the corpus represents and how invalid bytes/generation are scored. Hold labels
and evaluation results in independent custody; prohibit tuning after exposure.
A blinded fixed-prompt human assessment would need a predefined rubric, judges,
sampling plan and uncertainty reporting. This proposal is not authorization;
no external corpus was downloaded or evaluated here.

## Final selection and scientific limits

Keep three claims separate: B01 is a verified, about-half-F01-size candidate with
competitive selected validation BPB; E02 retains the smaller measured footprint
and lower training cost; none establishes reliable coherent language generation.
The shared BPE/scored bytes align likelihood comparisons, while capacity,
initialization, positions, depth, sessions and checkpoint-selection horizons
remain confounders. One seed provides neither variance nor significance.
Repeated architecture selection on the same validation suffix further limits
claims of unbiased generalization.

The predefined rule appropriately ended the authorized trajectory at4,500 and
preserved the selected3,000 state. No alternative patience rule, new seed,
additional architecture, original-test evaluation or extension was executed.
Engineering acceptance, complete original/recovered token reproduction and
durable evidence are documented in [B01_acceptance.md](B01_acceptance.md).
