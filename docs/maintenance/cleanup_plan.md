# Repository cleanup plan

Starting reviewed main: `cc7e345cee92c122cb1f7ad56a0d7627636934db`, merging
B01 PR20. Dedicated branch: `codex/repository-cleanup`. Baseline:1,276 tracked
files,94 script-directory files,20 notebook-directory files (15 actual ipynb),
29 test files,20 scientific modules and990 report files. Local full suite:
99 passed, one CUDA-only skip,14 dependency warnings in122.08s; pip check passes.

## Protected boundaries

Keep every path and byte under src/nano_lm, configs, experiments/configs and
assets/tokenizers. Keep all accepted report/evidence/qualification/failure records
and notebook/provenance bytes and paths; only report/README and notebooks/README
are mutable navigation pages. Preserve exact trained commits, run IDs, manifests,
hashes, canonical settings and original test policy. No dataset, checkpoint,
archive or external artifact modification/download, GPU allocation, research
training, API/frontend/deployment or final-submission notebook work.

The inventory reads every tracked file (binary evidence by identity/purpose),
parses all Python and notebook JSON, and searches Markdown, CI, notebooks and
runtime literals as well as imports. No file is deleted merely for lacking an
import. `baseline_integrity.json` records protected Git and local-checkout hashes
before changes; `python_dependency_review.json` catalogs all Python interfaces.

## Decisions to implement

| Change | Reason and compatibility |
|---|---|
| git mv handwritten-assignment.pdf → docs/handwritten-assignment.pdf | Preserve exact original PDF, place academic artifact beside assignment guidance; no runtime caller; historical Git-path mentions remain explicitly historical |
| git mv requirements-cpu-windows.lock.txt → docs/qualifications/requirements-cpu-windows.lock.txt | Preserve exact lock, co-locate environment records; current installation uses requirements.txt; only bootstrap inventory references the old root path |
| Remove data/.gitkeep | data/README.md already tracks the directory |
| Remove checkpoints/.gitkeep | No supported code reads this root placeholder; checkpoints are under unique run directories/external storage, future packaging may create parents |
| Remove outputs/{figures,logs,samples}/.gitkeep | Existing writers create parents; ignore the whole generated outputs tree, preserve existing local outputs and external artifact contents |
| Remove experiments/analysis/README.md | Consolidate duplicated collector guidance in docs/results_analysis.md, remove obsolete empty nesting, retain original text through starting commit |
| Share four bounded archive checksum copies and two identical sparse-snapshot reconstruction helpers | One small archive_tools module, original public names re-exported in historical scripts; keep streaming and rejection behavior, do not substitute whole-archive reads |
| Reuse identical E11/E12 token_budget helper | Keep both import interfaces; derive the same grouping, preserve invalid-divisibility failure and all numerical assertions |
| Rewrite current entry points/status pages | Concise README/project map/script and notebook catalogs; label chronological status/protocol records as historical, keep their bodies intact |
| Tighten .gitignore | Runtime outputs, caches, wheel/build files, temporary downloads and secrets ignored without hiding tracked scientific evidence |
| Retain two pre-existing broken hrefs in frozen F01 presentation guidance | The existing B01 parity guard prohibits even navigation edits there; current project map supplies correct report/generation links; regression checks reject every new broken source/target pair |

Do not move the scientific package/configs or research report directories. Keep
execution/qualification paths used by pinned historical notebooks and scripts.
The script families reflect materially different authorization gates, account
ownership, archive formats, interruption policies and acceptance contracts;
their similar filenames do not establish redundancy. Retain all unique report
writers and all15 notebooks, indexed by purpose rather than moved into legacy.
The CUDA-specific state roundtrip and source-qualified environment gate copies
remain unchanged: consolidation would add testing cost/risk without a useful
current supported-workflow reduction. No arbitrary target file-count reduction.

## Verification and delivery

Run the original suite before/after, regression checks for shared helper aliases,
streaming boundary hashes and corrupt snapshot rejection, editable install/pip
check, supported wheel and source-distribution builds, all primary imports/CLI
help, all 16 config counts and finite forwards, tokenizer loads, trusted fixture
checkpoint checks, notebook structure/code inspection and relative Markdown links.
Compare protected paths/bytes and both historical hash scopes. Audit ignore rules
and final staged native evidence. Preserve any pre-existing historical-link issue
explicitly if correcting it would rewrite immutable evidence; fix current links.

Record exact moves/deletions/consolidations, genuine failures, before/after counts
and intentionally retained/manual-review files in the changelog and verification
record. Push logical commits and open a reviewable PR; inspect actual CI success
and stop before merge. Future selected-model packaging, final paper, FastAPI,
frontend, hosted demonstration and presentation remain separately pending.
