# Supported tools and historical research workflows

The scientific implementation is [nano_lm](../src/nano_lm), not these wrappers.
Keep its stable source/config/tokenizer paths and hashes. This catalog separates
current offline interfaces from pinned historical cloud execution; no command
here authorizes new experiments, GPU jobs or test evaluation.

## Primary interfaces

| Task | Entry point |
|---|---|
| Acquire/check the pinned dataset | `python -m nano_lm.prepare --help` |
| Inspect configurations/benchmarks | `python -m nano_lm.experiments --help` — default audit covers E01–E14; maintenance verifies all 16 |
| Train/resume the one shared engine | `python -m nano_lm.train --help` — research requires new explicit approval |
| Conditional likelihood evaluation | Python API `nano_lm.evaluate.evaluate`; called by training/verification, no standalone CLI |
| Trusted-checkpoint inference | `python -m nano_lm.generate --help` |
| Export/verify/restore snapshots | `python -m nano_lm.storage --help` |
| Compatible pilot result collection | `python -m nano_lm.results --help`; use explicit incomplete-grid labeling |
| Single-run figures/numerical verification | `python -m nano_lm.plot_run --help`, `python -m nano_lm.verification --help` |

See [commands](../docs/commands.md), [BPB](../docs/bpb.md),
[storage](../docs/storage.md) and [recorded analysis](../docs/results_analysis.md).
The usual wheel contains the Python package; reproducible research also requires
the matching Git checkout/configs and separately verified dataset/checkpoints.
Standalone selected-model distribution is a later milestone.

## Artifact and analysis tools

- [pilot/download_kaggle_archive.py](pilot/download_kaggle_archive.py): authorized
  official streaming/range retrieval, explicit archive filename and whole SHA256.
- [pilot/inspect_archive.py](pilot/inspect_archive.py): strict pilot archive inspection.
- [final/inspect.py](final/inspect.py): F01 inventory/counters/sampler/CPU recovery.
- [balanced/inspect.py](balanced/inspect.py): B01 variable-stop inventory/counters/
  patience replay/sampler/CPU recovery.
- [final/report.py](final/report.py), [final/delivery.py](final/delivery.py) and
  [balanced/report.py](balanced/report.py): genuine receipt-gated figures/tables/report
  drafts; qualitative interpretation must be inspected. Write to fresh directories;
  never regenerate over accepted research evidence.
- [pilot/archive_tools.py](pilot/archive_tools.py): shared bounded checksum and
  exact sparse-snapshot reconstruction. Existing E02/E05/E07/E12 script names and
  imported helper symbols remain compatible.

These inspectors may deserialize trusted project states after integrity checks.
`--help` is safe interface inspection; it is not a completed recovery receipt.
Use each [research report](../report/README.md) for exact arguments/input schemas,
source versions, account access and existing retrieval evidence.

## Why historical families remain

| Directory | Purpose and boundary |
|---|---|
| [pilot](pilot) | E01–E14 specific mathematical audits, original narrow platform wrappers, interrupted-run reconciliation, comparisons and report writers |
| [final](final) | Separately approved F01 shape/10k/test policy, source guards, numerical references and exact recovery |
| [balanced](balanced) | B01's frozen stopping controller, no-refit data reuse, segmented-equivalence checks and variable-stop recovery |
| [kaggle_qualification](kaggle_qualification) | Historical account qualification; smoke-specific manifests/profile/state and genuine failure lineage |
| [maintenance](maintenance) | Offline inventory and source/evidence/link/notebook/interface checks |

Similar filenames do not imply interchangeable archive formats, model counts,
test policies, authorization gates or accounts. Legacy execution guards require
their recorded source commit/profile; do not remove those checks to run on current
main. Completed/failed notebook captures and accepted outputs remain at their
original paths. To reproduce a historical workflow, inspect its report and pinned
commit first; any new execution needs separate approval.

## Maintenance commands

```text
python -m scripts.maintenance.verify_repository --output outputs/cleanup/verification.json
python -m scripts.maintenance.repository_audit --help
```

The audit captures a baseline from a matching reviewed checkout before edits;
do not rerun it over the committed cleanup baseline to redefine preservation.
The verifier makes CPU finite-forward checks only: zero optimization, no corpus
download, no original-test likelihood and no external checkpoint loading. Existing
ordinary tests retain their trusted fixtures. [Inventory](../docs/maintenance/repository_inventory.csv)
and [changelog](../docs/maintenance/cleanup_changelog.md) describe every disposition.
