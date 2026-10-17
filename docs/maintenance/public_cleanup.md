# Public submission cleanup

Starting reviewed main: `8f88edee89dfceb6d0a8c84e385a886c4e058356`.
Branch: `codex/public-submission-cleanup`. No merge or history rewrite.

## Actual reduction

| Area | Before | After |
|---|---:|---:|
| Tracked files | 1,417 | 811 |
| Report directory files | 990 | 469 |
| Documentation files | 182 | 118 |
| Script directory files | 98 | 95 |
| Notebook directory files | 21 | 4 |

611 original files were removed and five purposeful files added: the preservation
contract, disposition ledger, shared historical-reference helper, this changelog
and the verification receipt. Net reduction: 606 files (42.77%). No relocations
were needed. Tracked Git-blob content decreased from 45.48 MB to approximately
21.58 MB; exact byte totals are in the [verification record](public_cleanup_verification.json).
This reduces the active checkout; historical Git objects remain intentionally
available and a normal full clone still transfers history.

## Removed and retained

[public_tree.csv](public_tree.csv) lists every original file, its Git blob,
classification, KEEP/ARCHIVE action and rationale. It is the complete deletion
mapping, not another active-tree inventory requiring manual refresh.

Removed from the active tree:

- Duplicate PNG/PDF plot exports and secondary per-run comparison plots. Retained
  canonical numerical inputs, comparison tables and selected loss/BPB PNGs.
- Intermediate CPU/development/preflight receipts; retained final artifact
  identities, environment/hardware, dataset verification, selected checkpoints,
  independent recovery, failure/replay accounting and timing evidence.
- Provisioning/research/recovery notebook captures and sidecar provenance files;
  retained the final clean submission notebook, accepted executed submission
  notebook and two generic package-importing research notebooks.
- Superseded qualification captures, readiness/progress/planning documents, old
  cleanup inventories and one-time qualification/inventory scripts.
- Intermediate F01 presentation-support exports; retained accepted analysis,
  interpretation, complete generations and final assignment evidence.

Every pilot keeps its individual analysis, summary, all twenty validation
observations, complete generation grid and necessary provenance/retrieval evidence.
F01 and B01 retain their canonical results, selected-model decisions and complete
interpretations. All sixteen definitions and scientific/tokenizer files retain
their original paths and bytes. Model packages remain external and unchanged.

Three qualification scripts and the superseded inventory generator were removed;
the historical-reference helper replaces the latter's shared read-only Git
utilities. The remaining script families are retained because tests, original
archive formats, numerical audits and checkpoint/figure reproduction depend on
them. For example F01 reporting imports the pilot prose formatter. Similar
filenames alone were not treated as proof of redundant implementation.

Two early receipts remain because retained research wrappers still read their
exact paths: `docs/evidence/dataset_manifest.json` and
`docs/qualifications/evidence/rayyan_fresh_gpu_recovery.json`. Full qualified
dependency freeze and the account qualification report also remain accessible.

## Preservation and dependencies

All 1,417 baseline Git blobs were checked before removal. The original commit is
still in public main history and anonymous retrieval of its old verification
receipt and historical notebook succeeded (HTTP 200). The ledger plus
[retrieval instructions](README.md) gives a path-specific route to every removed
file. This is existing public history, not a new upload or private archive. No
unresolved external archival destination or permissions decision was needed.

Retained research prose changes only archived Markdown link targets. Results,
interpretations and generated strings remain unchanged; the verifier checks the
exact allowed transformation against baseline Git objects. Current navigation
and historical execution guides were updated explicitly. No scientific imports,
configuration paths, checkpoint formats or tokenizer identities changed.

The old 1,103-file receipt remains immutable in history. The new verifier checks
577 retained protected artifacts, all sixteen configurations/tokenizers, four
remaining notebook files and current relative links, while separately checking
all 1,417 historical blob identities. The disposition ledger is itself
checksum-guarded; corruption cannot silently shrink the protection scope. Tests
exercise corrupt-ledger rejection and target-only link migration.

## Local output policy

Precise ignore rules cover virtual environments, caches, node_modules, Next.js
output, local secrets, downloaded datasets, model/checkpoint/archive files,
training journals, plotting histories, scratch notebooks, editor files and
render caches. They do not hide report/docs trees or all PNG/JSON/PDF files.
Existing generated article reviews remain ignored under outputs. Obsolete tracked
files were actually removed with Git; adding ignore rules was not used as a
substitute. Only the API/frontend example environment files remain tracked.

## Executed verification

- Baseline: 157 passed, one CUDA-only skip; original integrity 1,103 protected
  files/16 configs/17 notebooks. pip check passed.
- After: 160 passed, one CUDA-only skip; focused preservation/B01 suite 34 passed;
  fresh-clone scientific/submission/preservation suite 46 passed.
- Frontend: typecheck, seven tests and production build passed.
- Built wheel and source archive using installed setuptools.build_meta. The
  optional python-build frontend was unavailable; no dependencies were upgraded.
- Ten key submission graphics reproduced into ignored scratch storage from
  retained genuine measurements and explicit educational references.
- Fresh clean clone at `03c4004898a2a6357a3cd4c200516cf6e39ca2ef`: separately
  installed built wheel, asserted import location, anonymously downloaded and
  verified all six model-package files, loaded E02/B01/F01 and completed thirty
  generation cases. All twelve available raw-token reference cases matched.
  Genuine B01 API inference returned HTTP 200 and matched canonical decoding.
- Current-tree and fresh-clone verifiers pass with zero broken relative links.
  No internal Markdown heading-anchor links were found. No grandfathered link
  exceptions remain in the active tree. Source/notebook HTTP access was checked;
  no fresh Colab runtime or new research evaluation was launched.
- All forty frozen scientific/config/tokenizer checkout files match their exact
  pre-edit SHA256. Git diff contains no changes under those frozen prefixes.
- Credential-pattern scan of tracked files found zero matches. No model binaries,
  raw dataset copies or archive files are tracked. Removed files are untracked;
  legitimate ignored local outputs do not dirty the checkout.

The fresh clone reused installed Python 3.10.11 / torch 2.5.1+cpu dependencies; it
is not a new clean-dependency environment qualification. The baseline's first
pytest invocation encountered the existing shared Windows Temp ACL failure;
workspace-local --basetemp runs passed. Deprecation warnings remain visible.
Docker is unavailable locally; the unchanged CI container job must supply actual
container-build evidence. GPU verification was not executed and is not claimed.

The [machine-readable record](public_cleanup_verification.json) contains exact
hashes, counters, observed model identities, API receipt and verification limits.
Medium author approval/publication, instructor acceptance of HF hosting, recording
and class presentation remain separate human deliveries.
