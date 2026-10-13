# E14 execution, verification and operational evidence

Accepted on 2026-10-14. All four E14 saved notebooks completed: development,
development recovery, research, and final recovery. Only the research notebook
performed the authorized 5,000-step trajectory. No additional research seed,
configuration, hybrid or extension was launched.

## Source and account

Reviewed main: `2d698a671ebdae8e43d66e3062df174edd81cab0`, merged E13 PR #16.
Frozen trained source: `ab1da4e05005413807d20802fff4153d97ebfa10`.
Scientific hash: `cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
Git diff against E01's trained source showed no changes in scientific modules
or locked configurations. Source CI #38409926632 passed. Reporting helpers and
evidence have later commits; they were not copied uncommitted into training.

GitHub OAuth identity was verified as `rayyan-merchant`; Kaggle API identity as
`rayyannmerchant`. Existing scientific saved jobs were complete before allocation.
No E12 branch, artifacts or account credentials were used or changed. Each E14
phase had a unique private notebook ID and fresh paths. Quota before research:
24.23 GPU hours remaining; after completed phases: 23.67 hours remaining.

## Commands actually executed

```text
git fetch origin
git switch -c codex/e14-width240-seed42-20261013 origin/main
git diff 444bca4c4ac19f97f46fae210a9fd361f8561537 HEAD -- src configs experiments/configs
python -m pytest -q --basetemp D:/Nano-Language-Model-artifacts/E14_qualification_20261013/pytest_all_final
python -m pytest tests/test_e14_development.py tests/test_research_notebook.py tests/test_width_research.py tests/test_generation_grid.py -q
python scripts/pilot/width_audit.py --root . --experiment E14 --output CPU_width_audit.json
python scripts/pilot/build_notebook.py --experiment E14 --commit ab1da4e05005413807d20802fff4153d97ebfa10 --development-only --output DEVELOPMENT_SUBMISSION
kaggle kernels push -p DEVELOPMENT_SUBMISSION
python scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-e14-development-20261013-fa3959be/1 --manifest DEVELOPMENT_MANIFEST --output FRESH_DOWNLOAD --ranges
python scripts/pilot/inspect_development_archive.py DEVELOPMENT_ZIP DEVELOPMENT_MANIFEST FRESH_EXTRACT
python scripts/pilot/build_notebook.py --experiment E14 --commit ab1da4e05005413807d20802fff4153d97ebfa10 --source-kernel rayyannmerchant/nano-e14-development-20261013-fa3959be/1 --development-recovery --output DEVELOPMENT_RECOVERY_SUBMISSION
kaggle kernels push -p DEVELOPMENT_RECOVERY_SUBMISSION
python scripts/pilot/build_notebook.py --experiment E14 --commit ab1da4e05005413807d20802fff4153d97ebfa10 --qualification-kernel rayyannmerchant/nano-e14-development-recovery-20261013-c25df910/1 --output RESEARCH_SUBMISSION
kaggle kernels push -p RESEARCH_SUBMISSION
python scripts/pilot/build_notebook.py --experiment E14 --commit ab1da4e05005413807d20802fff4153d97ebfa10 --source-kernel rayyannmerchant/nano-e14-pilot-20261013-82e48728/1 --restore-only --output RECOVERY_SUBMISSION
kaggle kernels push -p RECOVERY_SUBMISSION
python scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-e14-pilot-20261013-82e48728/1 --manifest RESEARCH_MANIFEST --output FRESH_DOWNLOAD --ranges
python scripts/pilot/inspect_archive.py RESEARCH_ZIP RESEARCH_MANIFEST FRESH_EXTRACT
python scripts/pilot/diagnostics.py --artifacts VERIFIED --training-text data/processed/tiny-shakespeare-v3/train.txt --output checkpoint_diagnostics.json
python scripts/pilot/compare_heads.py --root . --experiment E14 --baseline VERIFIED_E01 --candidate VERIFIED_E14 --output comparison
python scripts/pilot/compare_widths.py --root . --e13 VERIFIED_E13 --baseline VERIFIED_E01 --e14 VERIFIED_E14 --output width_comparison
python scripts/pilot/compare_generation_grid.py --root . --experiment E14 --baseline E01_GENERATION --candidate E14_GENERATION --output generation_quality.json
python scripts/pilot/write_e14_report.py [verified artifact/comparison/recovery paths recorded in evidence]
```

Placeholders above shorten actual unique paths, not substitute implementations.
The exact research CLI arguments are preserved in `evidence/execution_command.json`.
Windows commands used `.venv/Scripts/python.exe`; current Kaggle API transfers
used the installed uv-tool Python 3.13. Cloud execution used isolated Python
3.11.14. Read-only status/log calls never resubmitted research.

## Actual test and integration evidence

- Initial full CPU suite: 68 passed, one CUDA-only skip, 193.45 seconds.
- Final full CPU suite: 69 passed, one CUDA-only skip, 116.89 seconds.
- Final focused orchestration/width/generation tests: 20 passed, 1.95 seconds.
- Focused recovery-gate checks: 15 passed, 3.57 seconds.
- Actual Kaggle development suite: 70 passed, 33.40 seconds.
- Actual Kaggle research preflight suite: 70 passed, 37.03 seconds.
- Twenty numerical width checks passed on CPU and actual qualified CUDA.
- Real Tiny Shakespeare CPU smoke and exact six-state roundtrip passed.
- Real CUDA smoke/full sliding validation/checkpoint/ZIP/Agg/generation passed.
- Published development read-back: all 81 files, snapshot, CPU inference and
  fresh exact CUDA state restoration at step 1, zero extra updates, passed.
- Research archive: all 62 files, CRC/safe paths/all SHA256 and best/final CPU
  inference passed. All 20 optimizer/scheduler counters and full histories passed.
- Both private samplers were independently replayed; final RNG and target-byte
  union matched E01. All 21 published snapshot manifests matched the native index;
  fresh CUDA recovery verified actual snapshot files and all state groups.
- Final state restored 5,000→5,000 with zero extra updates; best/final CPU/CUDA
  inference matched. All 100 generation slots passed unchanged protocol checks.

## Failures, corrections and evidence boundaries

Initial Windows test temporary-directory creation failed with access denied;
tests were rerun successfully against the authorized D-drive artifact location.
An initial CPU-smoke command used the parent data directory without a manifest;
the actual `tiny-shakespeare-v3` subdirectory was validated and its smoke passed.
An incorrect quota CLI subcommand was corrected to `kaggle quota`. A Git attribute
newline typo was corrected; original receipt bytes were explicitly verified
against committed blobs. These were preparation errors, not research results.

The original development and research jobs queued before running. Kaggle's log
service intermittently returned HTTP 500 or disconnected; local read-only
monitors reconnected and retained incident records. Duplicate log transport rows
with identical steps/metrics were deduplicated for progress. They were not replayed
optimizer updates. No duplicate research notebook was submitted.

The authoritative native journal contains 5,000 step events, no failure/resume
events, and no superseded attempts. Known physical research updates equal the
authoritative 5,000. All essential evidence is independently verified; no storage
failure or unresolved GPU recovery requirement remains for this individual run.
Final generation remains imperfect, and one seed does not resolve uncertainty.

## Time and memory scopes

Training 978.158 seconds; full validation 490.769 seconds. Checkpoint IO 3.516
seconds, exports 11.914 seconds, journal IO 18.735 seconds, generation 186.745
seconds, archive construction 34.087 seconds. These are separate measured scopes
and need not sum to pipeline time because preflight/bootstrap/other work exists.
The source-clone→archive-complete interval is 1,903.251 seconds; queue, notebook
publication, transfer and fresh recovery are outside it. Independent full research
download took 1,129.843 seconds, separate from GPU execution.

Recorded CUDA allocator peak is 177.283 MiB. Canonical telemetry resets once and
samples after updates, before validation. It includes earlier validation peaks,
not a separately measured final-validation/postprocessing peak. All three full
width runs share this policy. The short benchmark's final row precedes its only
full validation, so its memory reading is not substituted for research telemetry.
No FLOPs, energy, financial cost, queue duration or unrecorded timing was invented.


## Reviewed main integration at delivery

E12 PR18 merged during E14 execution at59f9d360875d3b29b6886378861b12de49b958f8.
It changes no scientific module or locked configuration. Reviewed main was merged
only into the E14 feature branch; the sole conflict in.gitattributes retained both
E12/E14 evidence-preservation intents. All85 imported E12 files matched reviewed
Git-index/working-copy bytes after normalizing ordinary Windows checkout line
endings. Combined delivery CPU suite:73passed,1CUDA-only skip,127.52s. The actual
E14 CUDA suite remains70passed on its frozen trained source; no extra GPU tests
or research were launched for the later reporting integration. PR17 is left for
review and is not merged automatically.
