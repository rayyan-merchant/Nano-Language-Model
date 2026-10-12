# Interpretation, limitations and reproducible recovery

This document accompanies [the measured report](E08_seed42_analysis.md).
All results are E08 seed42, one authorized 5000-update run. No other experiment,
seed or extension was launched. The research notebook and fresh recovery
notebook both completed their first saved version. Main was fetched again after
recovery and remained fad0afe2fc6259bd5fd2ddb5f004ce57e87bd34c. E06 PR9 had
already been merged by the owner; the review API returned no formal reviews.
This task did not merge it or claim a human review. E07 work is untouched.

## Hypothesis and observations

The prior hypothesis was that two additional blocks could change predictive
accuracy and optimization, while increasing capacity and computation. Improvement
was not assumed. The intended override is only n_layers:4 to6, but it adds
889728 trainable parameters (49.30%). This is a depth-plus-capacity comparison,
not a causal isolation of depth with parameter count held fixed.

**Observation:** E08 validation loss is lower at every one of the20 matched
scheduled checkpoints. Best loss is1.48713993 at4750, versus E01's1.53466801
at4750. The difference is-0.04752808 nats/token (-3.097%); best conditional
BPB differs by-0.06856853. Final differences are similarly negative, with
E08 BPB2.14758132 versus E01's2.21599704. These character experiments use
identical source bytes and scoring, so the comparison is directly meaningful.
BPB and token loss are mathematically linked here and are not independent wins.

**Interpretation:** Extra depth/capacity was beneficial for held-out next-character
likelihood under this one seed and fixed update/token budget. This does not
establish statistical significance, depth-only causality, transfer to other widths,
better understanding or improved free-running generation. Seed-level SD cannot
be estimated from one run. Across-update fluctuations are not seed uncertainty.

## Convergence and generalization

Observed loss decreases over1000-update intervals are0.21997119 (1k to2k),
0.11771208 (2k to3k),0.06035189 (3k to4k) and0.02857100 (4k to5k).
The3k-to5k improvement is0.08892289. Improvements are diminishing but have not
ceased. The best scheduled observation is4750; the final loss is only0.00145000
higher. There is one final uptick, not sustained deterioration. This is not
conclusive overfitting or double descent. Unknown between-evaluation minima
are not inferred, and no step-zero validation was performed.

The final100-update mean training loss is1.40996556, versus full sliding validation
loss1.48858993. Their gap is approximately0.07862438 nats. E01's corresponding
trailing-mean gap can be inspected in the genuine histories/plots; these sampled,
dropout-on training batches and dropout-off sliding validation have different
measurement conditions. A positive gap is a diagnostic, not a direct
memorization estimate. No test likelihood or test-based selection was used.

E08's lower likelihood loss costs more wall time. At3000 updates its training
time was906.855s and validation loss1.57751282, while E01 reached1.53601210
at5000 updates in916.708s. E08 first beat E01's final observed loss at3750
updates, after1134.011 training seconds. These retrospective comparisons use
recorded observations only; they are neither interpolated crossings nor controlled
equal-runtime experiments. E08 is more update-efficient here, but this evidence
does not demonstrate better time efficiency.

## Optimization and initialization

Recorded pre-clipping aggregate gradient norms were finite. E08 median0.49895141,
95th percentile0.54184342 and maximum2.16419029 compare with E01 median0.49782375,
95th percentile0.54196362 and maximum1.83285689. E08 clipped5 of5000 updates
(0.10%); E01 clipped7 (0.14%). E08's first/last100 mean norms were0.44049701
and0.51258392. Learning rate remained exactly0.0003 throughout. No numerical
failure or resume event occurred in the5000-step research journal.

These measurements do not suggest frequent clipping or an observed numerical
instability. They do not prove absence of vanishing gradients: research journals
record aggregate norms, not longitudinal per-layer gradients. Separate CUDA
development checks confirmed finite nonzero gradients in all six blocks; their
norms0.39845,0.38413,0.38671,0.37545,0.36991,0.35460 establish connectivity only.

Repeated seed42 construction was deterministic. Shared token embedding and the
first four blocks matched E01's initialization in the construction audit; the
output head weight did not, because two extra blocks consumed RNG draws before
its initialization. More dropout sites also change global training-RNG consumption.
The private data sampler's final RNG and byte-coverage arrays matched E01 exactly,
confirming comparable sampled source exposure, not identical optimization noise.

## Computational and storage cost

Training-only time was1511.357s (+64.868% versus E01); full validation consumed
593.279s (+40.952%). Training throughput was3.3083 updates/s or13550.73 targets/s.
Peak allocated GPU memory was166.104MiB, versus151.712MiB (+9.486%). This excludes
reserved memory, CUDA driver and whole-device usage. Both used TeslaT4 and the
same isolated dependency profile, but separate sessions leave hardware clock,
thermal and instance-load confounders. No timing significance is asserted.

Measured checkpoint calls took3.671s, snapshot exports10.657s and journal
flush/fsync17.583s. Full100-sample generation took280.250s; archive construction
and hash scope33.941s. The source-clone to archive-complete pipeline took2602.147s
(43.369 minutes). These scopes overlap the enclosing pipeline; do not sum them
again. Queue, notebook conversion/publication, network transfer and fresh-recovery
wall time are excluded. Local ZIP/hash verification plus CPU inference took51.751s;
this is a narrower measured recovery scope. No complete task lifetime or dollar cost was measured.

The measured33481866-byte smoke checkpoint informed a conservative5.496GB
free-space requirement;20.806GB was available at preflight. The final archive is
598768138 bytes. It contains all20 validation checkpoint files and complete
journals, metadata, generation and figures. Saved Kaggle outputs additionally
retain21 immutable snapshots. Large artifacts remain outside Git, on D drive
and in the private published notebook output. The source checkout remains on C.

## Genuine text generation and failure modes

[generation.md](generation.md) and [generation.json](generation.json) preserve
all100 outputs:50 settings each for best4750 and final5000. Each continuation has
256 new characters. All five prompts, greedy decoding, temperatures0.7/1.0/1.2
and sampling seeds42/123/456 are unchanged. These are actual CUDA-generated texts,
not externally edited samples. Five predetermined greedy examples are reproduced
in the main report, including their failures.

**Observation:** Best-checkpoint greedy continuations collapse into variations
of "The service of the world of the world". ROMEO and JULIET both start with
"And the world the world the seat of the world,"; the king/citizen prompts also
converge on similar words. This indicates weak prompt-specific characterization
and severe deterministic repetition. The best greedy mean repeated8-character
window-position fraction is0.62008, versus E01's0.52289. Final E08 is0.56225,
versus E01's0.54378. Lower E08 validation NLL therefore coexists with worse
observed repetition under this descriptive greedy diagnostic.

Stochastic outputs contain familiar speaker labels, blank-line dialogue formatting
and occasional plausible fragments, but frequent malformed words and broken syntax.
For example, best temperature0.7/seed42 JULIET begins "What is first your grace?"
then switches to LADY ANNE and later "The villate of an England's brother's present".
At temperature1.0 ROMEO includes "distand" and "contuant"; at1.2 First Citizen
contains "PrAULINA" and "O brun Exice". The king prompt often continues as
"The kingdom", which is character-continuation behavior rather than evidence of
semantic dialogue control. All poor samples remain in the grid.

Across15 best-checkpoint samples per temperature, repeated8-window fractions
average0.005087 (0.7),0.000535 (1.0) and0 (1.2). The final equivalents are0.007229,
0.000535 and0. Higher diversity is not equivalent to grammatical correctness.
The matching E01/E08100-row comparison is in [generation_quality.json](generation_quality.json).
Its fixed-window repetition statistic is a transparent descriptive heuristic,
not a validated language-quality score or independent replicated experiment.

The predeclared10-output overlap diagnostic (best greedy and temperature0.7/seed42
for all prompts) found longest exact training-text substrings of19 to22 characters.
They include ordinary phrases and speaker headers such as GLOUCESTER and Second
Servingman. This does not prove memorization or its absence; the check covers only
the stated subset and has no matched null distribution. No test-text overlap or
test likelihood selection was performed. Repetition loops themselves do not prove
training-text copying. Long-range consistency remains poor/unevaluated beyond the
short generated samples and128-character effective context.

## Actual tests and recovery evidence

Before launch: Windows full suite44 passed/1 CUDA-only skip in105.56s; focused
depth/notebook/grid tests6 passed in2.33s; CPU depth audit13 checks passed.
Actual Kaggle applicable full suite45 passed in33.84s, including CUDA AMP-resume
tests. A duplicated identical terminal result line is counted once. The research
run itself was float32, not AMP. Actual CUDA depth audit13 checks passed, plus
real-data one-update development training, validation, checkpoint save/load,
snapshot export/full-state roundtrip, ZIP CRC, Agg plotting and short generation.
Development steps were separate and excluded from the reseeded research history.
After reporting changes, the final Windows full suite passed44 tests with1
CUDA-only skip in174.53s. Final local checks, including figure/sample counts,
relative report links and preservation of prior science/results, are recorded
in [evidence/final_checks.json](evidence/final_checks.json).

The independent archive inspector verified exact archive byte size/SHA256, ZIP
CRC, safe paths, exact59-file set and every manifest file hash. It loaded best
and final for genuine CPU inference. A separate fresh saved Kaggle CUDA runtime
restored model, optimizer, scheduler, scaler, global RNG and sampling/training
state exactly at5000, and generated from both checkpoints with zero new updates.
The20 periodic states independently passed optimizer/scheduler/step-history audits.
Character BPB identity and55514 scored bytes passed at all20 observations.

The read-only local log monitor encountered intermittent ChunkedEncodingError
reconnections; this did not restart training. Platform notebook conversion printed
benign invalid-escape SyntaxWarnings after archive creation. Both saved notebooks
finished COMPLETE. The large download initially exposed a zero-byte file because
the Kaggle CLI buffers response.content before writing; after it completed,
all checks passed. These monitoring/download observations are not optimizer
interruptions. Authentic research journal failures/resumes are both zero.
The first local report-link audit found that its own final_checks.json receipt
had not yet been created; receipt creation order was corrected and the complete
audit rerun passed before commit. This reporting issue did not affect training,
evaluation, checkpoints or measured results.

### Durable retrieval

- Research saved output version1: [nano-e08-pilot-20261013-79fc614d](https://www.kaggle.com/code/rayyannmerchant/nano-e08-pilot-20261013-79fc614d).
- Fresh CUDA recovery version1: [nano-e08-recovery-20261013-75b12285](https://www.kaggle.com/code/rayyannmerchant/nano-e08-recovery-20261013-75b12285).
- Both are private; access requires the legitimately assigned rayyannmerchant account.
- Archive SHA256: `f1071463a41c18954cb8eb177b9e8a24ba9a692077ddb97e438c8af49d881fb4`.
- Local archive: `D:\Nano-Language-Model-artifacts\E08_published_20261013_79fc614d\E08_seed42_pilot.zip`.
- Verified extracted root: same directory's `verified` subdirectory.
- Checkpoints: `verified/runs/E08_seed42_a3d3b380c15940599e2efc2e08ac584a/checkpoints/step_004750.pt` (best) and `step_005000.pt` (final).
- Full checkpoint/snapshot manifest and hashes: evidence/artifact_manifest.json and evidence/snapshot_index.json.

Use fresh output paths, not an earlier experiment directory:

```powershell
kaggle kernels output rayyannmerchant/nano-e08-pilot-20261013-79fc614d/1 -p <fresh-download> --file-pattern 'E08_seed42_pilot.zip|artifact_manifest.json'
.venv\Scripts\python.exe scripts/pilot/inspect_archive.py <fresh-download>/E08_seed42_pilot.zip <fresh-download>/artifact_manifest.json <fresh-verified-root>
```

These are retrieval instructions, not new research-training authorization.
For compatible fresh GPU restoration use scripts/pilot/build_notebook.py's
restore-only mode pinned to trained source72fdd4b64610e8e6df178e7fd452ec9429faab8f
and the saved version1 source kernel, as recorded in the CUDA recovery receipt.
No10000-step extension has been started.

Analysis commands actually executed after retrieval use inspect_archive.py,
diagnostics.py, compare_depth.py, compare_generation_grid.py and
write_e08_report.py. [commands.json](commands.json) records their exact paths and
arguments plus the training command read from the published execution receipt.
They reproduce tables/figures from genuine checkpoints/journals; the narrative
interpretation is editorial and explicitly separate from measured data.

## Recommendation and remaining scientific limits

**Conclusion for this run:** Six blocks improve observed held-out character
prediction at the same5000-update budget, with substantially higher training
cost and retained generation failures. This is a promising capacity trade-off,
not evidence that six layers are universally optimal or depth alone caused it.
The separate E03 positional pilot has a lower recorded best BPB, but comparing
separate factors does not authorize or establish a combined architecture.

Keep5000 updates as the initial screening budget, with a proposal to prioritize
selected10000-update extensions only after explicit review/authorization. Continued
late improvement makes an extension informative, but does not make5000 a proven
convergence budget for all architectures. Replicated seeds and the pending E07
human-owned experiment matter more than declaring a winner from this one curve.
E09 is represented only as an unrun eight-block row with null metrics; it was
not executed. The broader three-seed study remains separately gated/NO-GO.

No train/evaluation/model/tokenizer semantics were changed. Core hash still matches
the E01 audited implementation. Reporting-only commits after the trained source
must not be substituted for its provenance. E08 PR12 is reviewable and is not
merged by this task. Prior experiment evidence is unchanged.
