# Interpretation, efficiency limits and reproducible evidence

This supplements [the measured E10 report](E10_seed42_analysis.md). All results
describe one explicitly authorized seed-42 run of 5,000 updates. No additional
configuration, seed, hybrid model or extension was executed. Hypotheses,
observations, interpretations and recommendations are distinguished below.

## Hypothesis and controlled architecture change

A smaller MLP might retain useful predictive quality with lower memory and
compute requirements, or lose useful capacity. No direction of effect was
assumed. E10 changes only mlp_expansion from four to two, hidden width768→384,
and removes591,360 parameters (32.765957%). Attention width192, six heads,
four blocks, context128, character tokenizer, GELU, sinusoidal positions and
dropout0.1 remain fixed. The explicit Q/K/V attention and pre-LayerNorm/residual
implementation is unchanged. This is an MLP architecture-plus-capacity ablation,
not an isolation of parameter count or a universal scaling-law experiment.

The 14 actual CPU/CUDA numerical checks verify all four MLP shapes and counts,
manual linear/GELU/residual/attention references, finite shifted CE and gradients,
future invariance, dropout and generation cropping. Changing tensor dimensions
changes default initialization scales where fan-in changes, RNG consumption and
subsequent initial weights; the initialization algorithm itself was not modified.
Smoke equality receipts explicitly record compatible-shape comparisons rather
than presuming whole-model equality under the same numeric seed. The private
data-sampling generator and final coverage arrays match E01 exactly.

## Predictive observations and interpretation

**Observation:** Best and final loss are 1.57144692, BPB
2.26711868, at5,000. E10 beats E01 at 1/20 matched
validation boundaries. Best BPB is 2.397%
above E01's best; final BPB is 2.307%
above E01's final. Exact absolute/relative deltas and all twenty measurements are
in comparison/matched_validation.csv and summary_comparison.csv. Character
loss/BPB are directly comparable over identical original scored bytes, but are
mathematically related, not independent evidence. Token perplexities are not
compared with BPE experiments. Test likelihood remains reserved.

**Interpretation:** Under this seed, data and update budget, reduced feed-forward
capacity/architecture has a predictive penalty. The magnitude should be weighed
against resource constraints; no preregistered tolerance defines “nearly the
same”, so the report does not declare equivalence. Nor is the model better simply
because it is smaller. One seed cannot establish significance, depth/width
generalization, parameter-count-only causality or absence of seed sensitivity.
Across-seed SD is undefined, not zero. Earlier depth/position experiments are
different single-factor comparisons, not evidence for an untested hybrid model.

## Convergence and generalization

| Observed interval | Validation NLL decrease |
|---|---:|
| 250–1000 | 0.44113143 |
| 1000–2000 | 0.21128968 |
| 2000–3000 | 0.12815174 |
| 3000–4000 | 0.06559836 |
| 4000–5000 | 0.04313092 |
| 3000–5000 | 0.10872927 |

**Observation:** Validation improves to the final scheduled checkpoint. The
3k–5k decrease is 0.10872927; the4k–5k decrease is
0.04313092.
The late observed-point fitted slope is -0.05532467
nats per1,000 updates. It is descriptive, not a forecast of a10k result. There
is no invented step-zero validation or inferred intermediate minimum.

**Interpretation:** Improvements diminish but have not ceased; the endpoint
does not demonstrate complete convergence. No sustained validation deterioration
is observed, so overfitting/double descent is not established. A smaller model
could have a capacity limit or simply need further optimization; this run does
not distinguish those hypotheses. Final100-update training mean is
1.53724579, versus validation
1.57144692, gap
0.03420113 nats.
Sampled dropout-on training and full dropout-off sliding validation have different
measurement conditions; this is a diagnostic gap, not a memorization rate.

## Efficiency: measured quality/resource tradeoffs

**Observation:** Training-only cost is 921.331s,
+0.504% versus E01. Twenty
full validations cost 315.430s,
-25.060% versus E01.
Peak allocated memory is 139.705MiB,
-7.915% versus E01. The smaller
parameter count does not deliver an observed training-speed improvement here.
Full validation is faster and allocated memory is lower. These are separately
measured scopes, not interchangeable throughput claims.

**Interpretation:** Predictive quality per update is worse, while model storage
and run memory requirements are lower. Quality versus training-only cost also
does not improve in this observation. The relative training difference is small
and was measured in separate T4 sessions, so it is not a robust timing advantage
or regression. We did not profile kernel utilization or CPU overhead; explanations
for the flat training time would be hypotheses, not measured causes. Evaluation
uses batch32 versus training microbatch4/accumulation8, so their measured effects
need not match. No isolated per-token deployment latency, FLOP count or monetary
cost was recorded. Generation-grid wall time includes loading and sampling and
is not an isolated inference benchmark.

The comparison stores an explicit marginal BPB penalty per million parameters
removed, paired quality/cost observations and targets per training second.
Dividing BPB by parameter count is not an intrinsic model-quality score. A compact
model can be useful under storage/memory constraints without being the best
predictor or fastest trainer. No resource weighting or equivalence tolerance was
predeclared, so no single arbitrary ratio is used to pronounce an overall winner.

## Optimization descriptors

All recorded losses and aggregate gradient norms are finite. E10 norm median is
0.49697378, p95 0.54495330, maximum 1.74943924; clipping
threshold1 was exceeded on 8/5,000 updates
(0.16%). AdamW, LR0.0003 constant, weight decay0.01,
float32 and effective4,096 target tokens/update remained fixed. All20 optimizer
and scheduler counters were checked independently. Per-layer longitudinal
research gradient traces were not recorded; finite nonzero smoke gradients in
each MLP do not prove absence of vanishing/exploding gradients throughout training.

## Genuine generation observations and failures

[All100 unedited outputs](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/pilots/E10_seed42/generation.md) retain five fixed prompts, both checkpoint
policies, greedy decoding, temperatures0.7/1/1.2, seeds42/123/456 and256 new tokens.
Best and final are the identical5,000-step state: their separately labeled50-row
grids duplicate the same settings, not independent model draws. The main report
shows all five predetermined best greedy examples, including poor outputs.

**Observation:** Mean repeated8-gram-position fraction for best greedy E10 is
0.529317, versus E01 0.522892.
All five greedy prompts converge toward “see”, “seem”, “sent” and “sense” loops.
ROMEO and JULIET responses show little persistent character-specific relevance.
Repetition obscures grammatical meaning despite line breaks and recognizable
local words. Temperature0.7 produces Shakespearean-looking speaker labels,
including KING RICHARD III and MENENIUS, without reliable syntax or narrative.
Higher temperatures avoid long exact loops but produce malformed words such as
“lapinish”, “reddistand”, “unorhice” and “kingdrat”. These examples come from the
fixed outputs; text was not rewritten or selected for favorable appearance.

Best temperature repetition fractions for0.7/1/1.2 are
0.002677/0.000000/0.000535.
Low exact repetition is not a grammar/coherence score. No qualitative superiority
or meaningful understanding is established. Identical sampling seeds can yield
shared trajectories under different prompts; the grid is not a set of independent
human-quality ratings.

Exact training-text overlap was checked on ten predetermined best outputs (all
five greedy and all five temperature0.7/seed42 samples). Longest matching
substrings range from 13
to 25 characters.
Exact strings and training offsets are in evidence/checkpoint_diagnostics.json.
Ordinary phrases repeat; this subset is not a comprehensive memorization audit
or proof that generated passages were copied. No test text was used.

## Preservation, failures and evidence boundaries

Research and fresh CUDA recovery saved version1 completed. Archive retrieval
passed full SHA256, exact entry set, ZIP CRC, safe extraction and every file hash.
Best/final CPU inference and fresh CUDA full-state roundtrip passed; model,
optimizer, scheduler, scaler, RNG, sampling and histories restored at5,000 with
zero additional updates. All20 checkpoints and21 immutable snapshots remain
published. The ZIP stores checkpoints once; duplicate snapshot files stay in
saved outputs. Large evidence remains on D: and Kaggle, outside Git.

The native research journal has no failure/resume event. Read-only SSE monitoring
required reconnects, without restarting training. Initial Windows test setup
failed due to a missing D-drive parent, then passed after directory creation.
The local generation comparison initially rejected E10 due to a reporting
allowlist omission; the guard and parameterized regression were fixed without
changing trained scientific code or outputs. A later default-temp permission
error was resolved by explicit D-drive basetemp. Two standard CLI archive
downloads ended in IncompleteRead; failed attempts were preserved. The unmodified
official Kaggle API supplied the authorized output URL. A streaming attempt
retained45MiB, then four bounded HTTP-range streams recovered the remaining
bytes. Exact206/Content-Range/size checks and full-archive SHA256 passed before
local CRC/all-file checks. An optional CPU-only fallback submission was rejected
by notebook-title validation before execution and was not retried after local
recovery succeeded; it used no compute. Exact receipts and commands
document these events, rather than presenting them as research restarts.

## Recommendation and optimized-model implications

Retain5k as the initial screening budget. Separately approved additional seeds
and selected10k extensions could distinguish convergence and seed effects, but
none was launched. E10's minimum at the endpoint and continuing3k–5k improvement
make extension informative; they do not prove it would match E01 or justify
silently changing the common budget.

Keep the4× MLP baseline when predictive quality under this training budget is
the priority. Retain2× as a compact candidate where the measured parameter/memory
reduction and validation cost matter enough to accept the observed likelihood
penalty. Additional seeds and explicitly defined resource/quality criteria are
needed before selecting an optimized architecture. Do not combine this change
with RoPE/depth/width or claim an untested hybrid gain. Single seed, one dataset,
finite context, initialization/dropout RNG differences, capacity and separate
GPU sessions constrain the conclusion. The broader study remains gated/NO-GO.

## Reproduction and delivery

[artifact_retrieval.md](artifact_retrieval.md) gives private saved-version URLs,
archive/checkpoint paths and retrieval procedures. evidence/commands.json records
actual execution/analysis commands; evidence/execution_command.json preserves
the unchanged training CLI. Final integrity receipts, tests and file hashes are
in evidence/final_checks.json and receipt_checksums.json. E10 PR14 is reviewable
and not auto-merged. E01–E09 records and the collaborator's E07 PR11 are preserved.
