# Reviewed interpretation and recovery record

## Observation

E03 has lower validation loss at all twenty matched evaluation steps. Final loss
is 1.43708178 versus E01's 1.53601210, an absolute decrease of 0.09893032
nats/token (6.4407%). Best loss improves by 0.09758623 (6.3588%). Final conditional
BPB decreases by 0.14272628. Both models have 1,804,800 parameters, the same
training-token budget, environment profile and scientific core code hash. Their
final sampling-generator state and complete target-byte coverage arrays match.
This supports a controlled positional-mode comparison, with one seed per model.

E03 first records loss below E01's final and best losses at update 2500
(1.52566536; 556.4783 accumulated training seconds). This is a retrospective
threshold among measured validation points, not an interpolated crossing or an
early-stopping experiment. Twenty correlated curve points are not twenty
independent replications.

Training-only runtime increases by 20.8368%; full-validation runtime increases by
6.3214%. Peak allocated memory is identical. RoPE's extra rotations offer a
plausible compute explanation, but no isolated kernel profiling or repeated
hardware timing trial establishes the cause. Better prediction per update does
not imply faster optimizer updates.

Validation improvements diminish: 0.15172 from 1k–2k, 0.07233 from 2k–3k,
0.03308 from 3k–4k, and 0.02960 from 4k–5k. A small increase of 0.000895 at
4000 follows 3750; every subsequent evaluation improves, reaching the observed
minimum at 5000. The curve has not flattened completely. There is no sustained
validation deterioration or defensible claim of double descent. Final validation
loss exceeds the last-100-update training mean by 0.07350, but training uses
dropout and sampled batches while validation uses evaluation mode and exhaustive
sliding scoring. This gap alone does not establish overfitting.

## Generation observations and negative results

The full fixed grid is retained. Best and final are the same update and weights;
the paired fifty outputs match exactly. They are not independent model samples.
All five greedy outputs enter formulaic loops around the prince, senators and
the sense. Repeated eight-character-position fractions are approximately
0.699, 0.823, 0.791, 0.546 and 0.739 in prompt order. E01's matching fractions
are 0.534, 0.410, 0.490, 0.574 and 0.606. E03 is more repetitive on four of
five prompts, despite its lower teacher-forced validation loss.

Temperature 0.7/seed42 outputs vary more and reproduce speaker labels and blank
line formatting, but contain grammatical errors and malformed words such as
"medding" and "elmost". Prompt responsiveness is limited; shared continuations
across different prompts remain visible. Higher-temperature examples are retained
in the unedited grid, rather than selecting only favorable outputs.

Exact training-text substring checks on the ten predetermined greedy and
temperature-0.7/seed42 outputs are recorded in checkpoint_diagnostics.json.
Short conventional phrases and speaker labels neither prove nor exclude
memorization. No held-out test likelihood or test-driven selection was performed.

## Interpretation and conclusion

For this seed, RoPE improves conditional next-token prediction at the same update
budget, at a measurable speed cost and without a greedy-generation quality win.
The experiment does not establish statistical significance, improved long-range
reasoning, context extrapolation, or superiority across architecture sizes.

Keep 5000 updates as an initial screening budget and prioritize separately
authorized 10000-update continuations of selected E01/E03 checkpoints and
additional seeds. Continued late improvement makes extensions informative;
diminishing improvements do not justify automatically increasing every model's
budget. No extension or other experiment was launched. E02 is outside this report.

## Failures, evidence preservation and retrieval

There was no research training interruption or nonfinite-loss failure. Original
notebook and fresh recovery notebook version1 both reached COMPLETE. Local C-drive
extraction initially failed with errno28. The published archive remained intact;
after rechecking its SHA256, only the incomplete extraction was removed. A new
unique D-drive location holds the archive and complete independently verified
recovery. E01 artifacts were unchanged. CPU inference and fresh Kaggle CUDA
state restoration subsequently passed. This incident affected local recovery,
not training or publication. Receipts preserve the failure and resolution.

Archive: 402094081 bytes; SHA256
`57feb9d606a1d733ce3ae301b9e401c12b200f32c1d3c3fca4c47c3c6117b110`.
All 58 files passed CRC, safe-path, file-set and per-file checksum verification.
All 21 snapshot manifests and all 20 checkpoint counters were verified.
Best/final checkpoint SHA256:
`d251d9b2e2d5b30bda3cf6db3a6e88ce37e8a97c5b778646c4d2ea9eca8d7f0a`.

Authoritative local root:
`D:/Nano-Language-Model-artifacts/E03_seed42_3eddd727e88a4748b23b605c7c3590fd/verified`.
The checkpoint is under
`runs/E03_seed42_3eddd727e88a4748b23b605c7c3590fd/checkpoints/step_005000.pt`.
The same artifact is recoverable from the private published notebook version1:
https://www.kaggle.com/code/rayyannmerchant/nano-e03-pilot-20261012-82f33b8a
Fresh recovery proof:
https://www.kaggle.com/code/rayyannmerchant/nano-e03-recovery-20261012-33b2c3ba

Download into a new empty location using the legitimate account:

```powershell
kaggle kernels output rayyannmerchant/nano-e03-pilot-20261012-82f33b8a/1 -p NEW_DOWNLOAD_DIRECTORY
python scripts/pilot/inspect_archive.py NEW_DOWNLOAD_DIRECTORY/E03_seed42_pilot.zip ARTIFACT_MANIFEST_JSON NEW_EMPTY_RECOVERY_DIRECTORY
```

Use the archive inspector's documented arguments and a fresh extraction directory;
do not blindly trust downloaded pickle checkpoints. The committed evidence manifest
contains the complete archived file list and hashes. Large checkpoints, dataset,
full step journal and snapshot copies remain outside GitHub.

Measured full original pipeline time is 1951.87148 seconds, excluding queue,
publication conversion and independent retrieval. Independent download scope was
259.37044 seconds. Fresh recovery has no explicit end marker, so its exact pipeline
duration is unmeasured, not inferred from log timestamps. Publication overhead and
dollar costs are not measured. Restoring the float32 disabled scaler is verified;
this research experiment did not test AMP or perform an extra training update.

## Actual verification scope

The pre-execution Windows suite passed 36 tests with one CUDA skip in 206.61 s.
The clean training commit's GitHub CPU CI succeeded before submission. Thirteen
RoPE-specific numerical checks passed on CPU and actual Kaggle CUDA. A genuine
one-update CUDA development run exercised reload, export, nine plots and shortened
generation before separately reseeded research training. Development measurements
are not research observations. Fresh published-output recovery exactly restored
model, optimizer, scheduler, disabled scaler, RNG and sampling/training state at
5000 with zero additional updates and actual GPU generation. No claim is made that
the entire Windows pytest suite was rerun on Kaggle for E03.

Executed entry points: `pytest -q`; `scripts/pilot/rope_audit.py`;
`scripts/pilot/build_notebook.py`; canonical `python -m nano_lm.train` through
`scripts/pilot/run.py`; `scripts/pilot/analyze.py`; `scripts/pilot/capture_platform_log.py`;
`scripts/pilot/inspect_archive.py`; `scripts/pilot/recover.py` in a fresh CUDA
notebook; `scripts/pilot/compare_positions.py`; `scripts/pilot/write_e03_report.py`.
Resolved research command, configuration, source, environment and preflight
arguments are preserved in evidence/execution_command.json and the published
notebook source metadata. The architecture/configuration/scientific source was
not changed. Engineering review and passing tests are documented; no human PR
approval is represented. PR6 remains reviewable and is not merged.
