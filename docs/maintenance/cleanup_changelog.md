# Cleanup relocation, deletion and consolidation record

Starting reviewed main: `cc7e345cee92c122cb1f7ad56a0d7627636934db`.
Original versions remain recoverable with `git show BASE:PATH`. No force-push,
history rewrite, research retraining or accepted artifact regeneration.

## Complete relocation/deletion map

| Original tracked path | Action / replacement | Safety |
|---|---|---|
| handwritten-assignment.pdf | git mv → docs/handwritten-assignment.pdf | Exact original PDF preserved; no runtime caller; old historical Git-path mentions retained |
| requirements-cpu-windows.lock.txt | git mv → docs/qualifications/requirements-cpu-windows.lock.txt | Exact lock preserved; current install uses requirements.txt; historical foundation inventory retained |
| data/.gitkeep | Removed; data/README.md retains directory | Redundant empty placeholder |
| checkpoints/.gitkeep | Removed; no tracked root checkpoints directory | Runtime states belong to unique run directories/external storage; no code needs this placeholder |
| outputs/figures/.gitkeep | Removed | Writers create parents; existing generated figures untouched |
| outputs/logs/.gitkeep | Removed | Writers create parents; existing generated logs untouched |
| outputs/samples/.gitkeep | Removed | Writers create parents; existing generated samples untouched |
| experiments/analysis/README.md | Removed; consolidated into docs/results_analysis.md | Duplicate collector guidance/stale status; no runtime, CI or notebook dependency; original bootstrap path remains in immutable changed_files inventory |

Only empty local obsolete directories may be removed after checking their contents
and resolved workspace paths. Never delete or relocate the existing outputs directory or any outputs junction, prepared dataset
cache or any external artifact folder to simplify the tree.

## Shared code and public compatibility

Four bounded checksum copies in verify_e02/e05/e07/e12_archive and two identical
E07/E12 sparse-snapshot reconstruction helpers now live in
`scripts/pilot/archive_tools.py`. Existing historical script paths, `checksum`
and `reconstruct_snapshots` imported names remain available, with both direct
script and package execution supported. Streaming remains1MiB per read; sparse
checkpoint bytes/manifests and corrupt-hash rejection remain unchanged.

`scripts/pilot/e12_context_audit.py` re-exports the identical
`context_audit.token_budget` implementation. E01 context 128/accumulation 8,
E11 context 64/accumulation 16 and E12 context 256/accumulation 4 still produce 4096
positions. The retained `audit`/`verify` bodies are AST-identical to reviewed main.
Numerical assertions were not loosened, and existing tests were extended to check
both reconstruction entry points rather than removed or skipped.

No historical presentation file or producer was changed. An attempted correction
of two existing relative hrefs in F01 findings_and_limitations.md triggered the
original B01 immutable-evidence parity test and was reverted, including the
producer change. The original parity guard and all numerical assertions remain
intact. Current project-map links point directly to the valid report and generation
destinations. The link checker reports those exact two unchanged historical
defects and rejects every new broken source/target pair; it does not ignore whole
evidence directories. No measurement, figure, output or manifest was regenerated.

## Documentation and ignored outputs

Root README, docs/results_analysis, report/README and notebooks/README are current
navigation pages. New docs/README and scripts/README catalog stable interfaces,
all 16 configurations and 15 notebooks, accepted reports and pending delivery work.
AGENTS protects frozen source/config/tokenizer paths and distinguishes completed
single-seed research from the discontinued grid and unimplemented deployment.
Current-status headers were added to readiness, progress, compliance, protocol,
experiment design, commands, platforms, storage, collaboration, decisions and 15
execution guides; their original chronological/procedure bodies remain intact.

The evaluator is a Python API, not a standalone CLI. Running its module with
`--help` silently exits without help; current docs and verification do not claim
that as a working evaluation CLI. New maintenance checks validate genuine argparse
help, imports, configuration counts, tokenizers, notebook structure and file links.

.gitignore now covers all generated outputs/data except data/README, root generated
checkpoints, binary model/archive/download files, notebook/cache files and local
secrets; it does not ignore research plots, handwritten PDF, canonical artifacts
or historical notebooks. No dependency pins were changed.

## Intentionally retained / manual review

All 990 report files and unique notebook/provenance records are retained; their
size reflects genuine results, graphs, fixed generations, audits and failures.
Legacy execution families retain distinct source/account/auth/scoring/replay gates.
The CUDA state-roundtrip and exact environment-gate duplicates remain untouched:
removing them would add risk without a needed supported-workflow reduction.
`docs/evidence/verify_gpu_export.py` is a historical fixed-local-layout verifier;
retain it for provenance/manual review, not as a portable current entry point.
The handwritten calculation is preserved, not newly certified correct.
The two F01 presentation href defects require a separately authorized evidence
update if their original bytes are to change; they do not block current navigation,
source inference, artifact retrieval or checkpoint compatibility.

Final counts, build/import/link/CI outcomes and any limitations are recorded in
[cleanup_verification.md](cleanup_verification.md). A new developer should read
root README → docs/README → scripts/README, then the relevant accepted report and
AGENTS before changing or executing anything.
