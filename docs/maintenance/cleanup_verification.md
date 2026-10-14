# Repository cleanup verification

Cleanup is verified locally against reviewed main
`cc7e345cee92c122cb1f7ad56a0d7627636934db`. This is an offline maintenance
audit, not a new research qualification or final submission acceptance.
Delivery is [cleanup PR #21](https://github.com/rayyan-merchant/Nano-Language-Model/pull/21),
left open for review rather than merged. Linux CPU CI on source
`f49cb3ad699b89807ba17f1394bb09b3d8c054fc` passed in
[workflow run 38620540504](https://github.com/rayyan-merchant/Nano-Language-Model/actions/runs/38620540504):
121 passed, 1 CUDA-only skipped, 14 dependency warnings in 26.70 s. Its subsequent
offline verifier also passed all 1,103 protected files, 16 configurations and 15
notebooks, reporting the two unchanged historical link defects. This CI receipt
documents that exact source; the PR checks cover subsequent documentation commits.

## Before and after

| Inventory | Before | After |
|---|---:|---:|
| Repository files | 1,276 | 1,285 |
| Files under scripts | 94 | 98 |
| Python scripts | 94 | 97 |
| Notebook directory files | 20 | 20 |
| Actual ipynb notebooks | 15 | 15 |
| Report files | 990 | 990 |
| Test files | 29 | 30 |
| Scientific package modules | 20 | 20 |
| Root files | 9 | 7 |

Two files were relocated using `git mv`; six redundant files were removed.
Fifteen purposeful files were added for navigation, the shared archive helper,
regression tests and the requested audit records. This increases the total file
count while reducing root clutter and duplicated implementation. The complete
mapping is in [cleanup_changelog.md](cleanup_changelog.md); the inventory retains
all removed rows as well as every resulting repository file.

Consolidation removes 39 duplicated function lines: three extra checksum
copies (15 lines), one sparse reconstruction copy (15 lines), and one token-budget
copy (9 lines). Public historical imports and direct script commands remain
available. Environment-specific CUDA recovery gates were deliberately retained.
The 990 report files contain unique accepted evidence, not disposable clutter.

The supported structure is root guidance and packaging, stable `src/nano_lm`,
`configs`, `experiments/configs` and `assets/tokenizers`, execution families under
`scripts`, curated historical `notebooks`, tests, accepted `report` evidence and
indexed `docs`. `data/README.md` retains dataset guidance. Generated outputs and
checkpoints are ignored and created by their writers; `api` remains reserved.
No empty tracking placeholders are needed for generated directories.

## Commands actually executed

Commands used the existing Windows `.venv/Scripts/python.exe` (Python 3.10,
PyTorch 2.5.1 CPU); no dependency pins or installed research versions changed.

| Check | Actual outcome |
|---|---|
| `python -m pytest -q` before changes | 99 passed, 1 CUDA-only skipped; 14 dependency warnings; 122.08 s |
| Focused cleanup, reconstruction and context regression tests | 30 passed; 39.40 s before the final extra negative link test |
| First complete cleanup test attempt | 119 passed, 1 failed, 1 skipped; 135.27 s; preservation failure described below |
| Corrected `python -m pytest -q` | 121 passed, 1 CUDA-only skipped; 14 Pyparsing dependency warnings; 135.97 s |
| `python -m pip check` | No broken requirements |
| `python -m pip install -e . --no-deps --no-build-isolation` | Editable package installation succeeded |
| `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir outputs/cleanup_20261014/builds` | Wheel built successfully |
| `setuptools.build_meta.build_sdist('outputs/cleanup_20261014/builds')` | Source distribution built successfully |
| Wheel installation with `--target outputs/cleanup_20261014/wheel_install --no-deps --no-compile` | Succeeded; all 20 package modules imported from that independent target |
| `python -m scripts.maintenance.verify_repository --output outputs/cleanup_20261014/verification_final.json` | Protected-file integrity, imports, configurations, tokenizers, notebooks and link regression checks passed |
| `python -m scripts.maintenance.repository_audit --refresh-current` | Current inventory coverage verified without redefining the baseline |

The wheel is 31,301 bytes, SHA256
`9624e403f3696b8884b2932ccc4d5c29ba76454e15b39a99515295b1166c3cd3`.
Builds used the existing declared setuptools 75.6.0 backend; the optional `build`
frontend was unavailable. Builds remain ignored local outputs. These distributions
package the Python library, not a self-contained research delivery: reproduction
still requires the matching Git checkout for configurations, tools and evidence.
Final model/reproducibility packaging remains subsequent work.

Machine-readable receipts: [verification_receipt.json](verification_receipt.json)
and [wheel_imports.json](wheel_imports.json). The baseline dependency/AST review
and file reference inventory are recorded beside them. Baseline references describe
the original tree; newly created rows include current references.

## Scientific and compatibility evidence

All 1,103 protected source/configuration/tokenizer/research/notebook paths retain
their original Git-normalized bytes and, on Windows, their original raw checkout
hashes. Both relocated files retain original content. No scientific path moved.

| Fingerprint scope | Unchanged value |
|---|---|
| Historical E01–E14 audited source/configuration set | `cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6` |
| Current aggregate including F01/B01 configurations | `4dd230985194865487254e2e750b07f79aeabec491e4fa8f8e483fedda1de741` |
| Scientific modules only | `d5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8` |

These scopes differ because combined architecture configurations were added before
cleanup, not because cleanup changed training semantics. All 16 configurations
instantiate with recorded parameter counts and finite CPU forward outputs.
Both canonical tokenizer artifacts load and round-trip known text with their
expected semantic hashes. All 15 notebooks retain their original bytes and pass
JSON/cell checks; Python cells compile, while IPython magic cells receive structural
checks. No notebook or remote GPU job was executed.

Nine genuine CLI help interfaces and all 20 package imports were checked. Evaluation
is the existing `nano_lm.evaluate.evaluate` Python API: the module has no standalone
CLI, so a silent `--help` exit is not counted as a supported command.
The existing suite exercises checkpoint/RNG restoration with trusted fixtures;
cleanup added shared-helper identity, bounded hashing, corruption rejection,
direct-script invocation, token-budget and ignore/link regression tests.

Original document bodies remain intact beneath clearly labeled current-status
headers. Accepted results, source identities, archive hashes and artifact retrieval
records were preserved. No large archive was downloaded or republished, and private
artifact availability was not freshly requalified. Existing local outputs and
external artifact folders were untouched apart from redundant tracked placeholders.
The current local `outputs` directory is an ordinary directory on C:, not a D:
junction; moving storage is outside this cleanup.

## Failures and remaining limitations

An attempted correction of two relative links in F01 presentation evidence failed
the original B01 historical-parity test. Both the guide and its producer were
restored; no guard or scientific assertion was weakened. Current navigation links
directly to the correct files. The checker reports only these two original defects
and rejects any new broken source/target pair:

- `report/final/presentation_support/findings_and_limitations.md` → `F01_10000_analysis.md`
- The same guide → `generation.md`

Changing those frozen evidence bytes requires separate evidence-update review.
Relative file links are checked; heading fragments and live HTTP/private artifact
availability are not. The CUDA-only skip is explicit: there was no GPU allocation.
`docs/evidence/verify_gpu_export.py` retains fixed historical local paths for manual
review. The handwritten assignment PDF was relocated intact, not newly certified.

There was no retraining, dataset acquisition, original-test likelihood evaluation,
checkpoint artifact rewrite, result regeneration, API, frontend or deployment work.
The cleanup PR must be reviewed before merge. Final academic reporting, standalone
inference packaging and deployment remain subsequent deliverables.

For a new developer: read the root README, [project map](../README.md),
[script catalog](../../scripts/README.md), AGENTS and the relevant accepted report
before modifying code or requesting any further execution.
