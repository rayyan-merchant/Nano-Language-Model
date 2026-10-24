# Interpretation, scientific limits and reproducible recovery

This accompanies [the measured E09 report](E09_seed42_analysis.md). The report,
tables and figures describe one authorized E09 seed-42 run, not a multi-seed
result. No other research configuration, seed or extension was launched.

## Hypothesis, observations and interpretation

The prior hypothesis was that additional Transformer blocks might change
predictive accuracy and optimization, while increasing capacity and cost.
Improvement was not assumed. Only n_layers changes from four to eight; the
model adds 1,779,456 parameters (98.5957%). Compared with six-block E08 it adds
889,728 parameters. Width, tokenizer, context, optimizer, token budget,
validation and precision are identical. This isolates the configuration override,
but does not isolate depth independently of capacity or computation.

**Observation:** E09 has lower validation loss at 20/20 matching
checkpoints versus E01 and 20/20 versus E08. Best loss is
1.45571123 at update 4750; best
conditional BPB is 2.10014737. Best BPB changes by
-5.145%
against E01 and -2.113%
against E08. All twenty exact matched observations and absolute/relative deltas
are in comparison/matched_validation.csv and comparison/pairwise_differences.csv.

**Interpretation:** Additional depth/capacity benefits held-out next-character
likelihood for this seed and update budget. Character BPB is loss divided by
ln(2), so lower NLL and BPB are related evidence, not independent achievements.
No statistical significance, depth-only causality, general superiority or
improved understanding is established. One run per model cannot estimate
across-seed SD; fluctuations across optimizer updates are not seed uncertainty.

## Convergence and generalization

The first training loss is recorded at update 1, the first validation at 250.
There is no invented step-zero validation. Observed interval improvements are:

| Interval | Validation NLL decrease | Decrease per 1,000 updates |
|---|---:|---:|
| 250–1000 | 0.54573457 | 0.72764610 |
| 1000–2000 | 0.22513656 | 0.22513656 |
| 2000–3000 | 0.09447299 | 0.09447299 |
| 3000–4000 | 0.06518158 | 0.06518158 |
| 4000–5000 | 0.02665455 | 0.02665455 |
| 3000–5000 | 0.09183613 | 0.04591806 |

**Observation:** The net 3k–5k loss improvement is
0.09183613. Improvement is diminishing over successive
1,000-update intervals, but has not ceased. The best scheduled point is 4,750;
the final loss rises by 0.01154868
(0.793%). The final
loss remains below the 4,500 measurement. Minima between scheduled evaluations
are unknown. The late observed-point fitted slope is descriptive, not a forecast.

**Interpretation:** One final uptick does not establish sustained overfitting or
double descent. The trailing 100-update training mean is
1.36598722, giving a diagnostic final validation
gap of 0.10127268 nats.
Training uses sampled windows with dropout; validation uses complete sliding
windows without dropout. The gap therefore is not a direct memorization rate.
It is larger than E08's corresponding gap, but one-seed uncertainty remains.
No test likelihood or test-based architecture selection was performed.

## Measured cost and compute efficiency

E09 uses 1811.126 training-only seconds,
739.049 full-validation seconds and
199.370 MiB peak allocated GPU memory. Training time changes
by 97.568% versus E01 and
19.834% versus E08. Parameter
count and actual training time are distinct from the fixed 20,480,000 target-token
budget. All three final private sampling RNG states and target-byte coverage
arrays match. Repeated target exposure is not independent examples or epochs.
Only 1,003,848 of 1,003,854 training bytes were covered as targets.

At E01-like training cost, E09's observed 2,500-step point used 907.084 seconds
and had loss 1.58724586, while E01 ended at loss 1.53601210 after 916.708 seconds.
These are actual nearby observations, not an interpolated crossing or a controlled
equal-runtime experiment. Extra depth is more effective per update here, but
does not establish better predictive performance per GPU second. The nine
comparison plots include BPB versus updates, cumulative measured training time,
parameter count and allocated memory. Separate T4 sessions and execution noise
limit speed comparisons. Allocated memory excludes reserved/driver/device costs.
No dollar cost or FLOP measurement was recorded.

The aggregate pre-clipping gradient median is 0.49622917, p95
0.53815512, maximum 2.39997935; clipping threshold 1 was
exceeded on 4/5,000 updates
(0.08%). Norms and losses are finite throughout.
These are aggregate gradients; per-layer longitudinal research gradients were
not recorded. CPU/CUDA smoke connectivity across all eight blocks does not prove
the absence of vanishing/exploding gradients during all research updates.

## Complete genuine generation evaluation

[All 100 unedited outputs](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/pilots/E09_seed42/generation.md) preserve both checkpoint policies,
five fixed prompts, greedy decoding, temperatures 0.7/1.0/1.2, seeds 42/123/456
and 256 new characters. The measured report shows all five predetermined best
greedy samples, including their failures. No external model edited these outputs.

The fraction of repeated 8-gram positions averages 0.330120 for
E09 best greedy outputs, versus E01 0.522892 and E08 0.620080. Final E09 greedy
averages 0.486747, versus E01 0.543775 and E08 0.562249.
These descriptive fixed-grid values suggest less exact repetition in E09's best
greedy grid than E08's, but do not establish coherent language or a general
decoding improvement. Best temperature means are
0.002677/0.003481/0.001071
for 0.7/1.0/1.2. Low exact repetition does not imply correct grammar.

**Observed failures:** Best greedy prompts repeatedly converge to “Second
Murderer”, “What shall he will have the present of the prince?” and repeated
“so”. JULIET produces “the seat of the seat” and “the season of the season”.
Dialogue labels, line breaks and familiar local words are recognizable, while
semantic continuity, grammar and prompt-specific character behavior remain weak.
Temperature sampling reduces long exact loops but produces malformed words such
as “modtal”, “promper”, “meddital”, “latements” and “pospering”. Samples with
speaker labels and less repetition still often have inconsistent syntax and
little longer-distance meaning. These are illustrative observations from the
fixed grid, not an automatic coherence rating or statistically significant gain.

Exact training-text overlap was checked for ten predetermined best samples
(all five greedy and all five temperature-0.7/seed-42 outputs). Their longest
matching substrings range from 14
to 32 characters.
See evidence/checkpoint_diagnostics.json for exact strings and offsets. Ordinary
Shakespearean phrases repeat; this subset diagnostic is not a comprehensive
memorization audit or proof of copied training passages. No test text was used.

## Preservation, recovery and failures

The research saved notebook version 1 completed. The 794,894,378-byte
archive passed SHA256, exact file set, safe extraction, ZIP CRC and all 59 file
hashes independently. Best/final CPU inference passed. Fresh compatible CUDA
recovery `rayyannmerchant/nano-e09-recovery-20261013-66f79ad5` restored model, optimizer, scheduler, scaler, RNG,
sampling and history exactly at 5,000, with zero additional updates. Its recovered
best/final checkpoint hashes match the archive manifest. All twenty optimizer
and scheduler counters were independently checked. Twenty-one immutable snapshots
are retained in published `pilot/snapshots`; the ZIP omits their duplicate files
but includes every resumable run checkpoint plus the snapshot index/manifests.

Large files are on D: and in saved Kaggle outputs, excluded from Git. The local
verified run is `D:\Nano-Language-Model-artifacts\E09_published_20261013_b9ea5610\verified`. Checkpoint hashes and retrieval commands are recorded
with the manifest and executed-command evidence. Both published versions are
private to the legitimate rayyannmerchant account.

The local read-only log stream required reconnects and temporarily returned 401
after OAuth access expiration. Official existing-account OAuth refresh restored
access; the user also confirmed private reconnection. These were monitoring
failures, not trainer restarts. No research failure or resume event appears in
the 5,000-step journal. No credential, installed SDK patch, quota workaround or
uncommitted Windows training code was used. Headless plotting and mounted
read-only recovery inputs worked; the earlier E01 plotting issue did not recur.

## Recommendation and remaining limits

Keep 5k as an initial screening budget, with separately approved 10k extensions
of selected configurations considered informative. E09 improves through 4,750
and has a nontrivial 3k–5k gain, so 5k does not prove convergence. The final uptick
and generation loops make a longer controlled comparison worth investigating,
not automatically better. First prioritize additional independently authorized
seeds and matched provenance; no extension is launched by this recommendation.

E03 RoPE's contextual best BPB 2.07327076 at 1,804,800 parameters remains better
than this E09 observation. It is a separate positional ablation, not a depth
control or evidence for an untested combined architecture. Single dataset,
single seed, extra capacity, initialization/dropout RNG consumption, finite
context and separate GPU-session costs constrain every conclusion. The first
four block/token initialization weights match E01, while the output head changes
because extra block initialization consumes RNG. Identical numeric seeds do not
mean identical whole-model initialization. Across-seed SD remains undefined.

The authorized E09 run is complete; the broader three-seed study remains
separately gated/NO-GO. Main and all earlier experiments are preserved. E08 PR12
was owner-merged before preparation; no formal review submission was returned
and none is claimed. This task opens/updates E09 PR13 for review and does not
merge it. E07's independent work is untouched.

## Reproduction evidence

- Cloud training: evidence/execution_command.json, using unchanged shared CLI.
- Source and actual CPU/CUDA tests: evidence/execution_review.json and audits.
- Local verification/analysis commands: evidence/commands.json.
- Final report integrity checks: evidence/final_checks.json.
- Exact archive file hashes: evidence/artifact_manifest.json.
- CPU recovery: evidence/independent_recovery.json.
- Fresh zero-update CUDA recovery: evidence/cuda_recovery.json.
- Measured timing scopes: evidence/platform_timing.json and evidence/io_timing.json.
- Retrieval instructions: [artifact_retrieval.md](artifact_retrieval.md).

All figures and tables derive from actual recorded histories. No unobserved
validation point, synthetic research measurement, fabricated SD or successful
unexecuted test is presented.
