# Public tree integrity and historical retrieval

## Internal-material cleanup — 2026-10-24

397 internal-only files were removed from the current public tree. Exact paths,
Git blob IDs, sizes and reasons are in [internal_archive.csv](internal_archive.csv).
Their immutable source is commit `5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430`;
use `git show 5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430:PATH` or the pinned links
in retained reports to inspect historical receipts. A checksum-verified local
backup is stored outside Git in `outputs/internal-archive-20261024.zip`.
Exact anchored ignore rules prevent these operational records from being
accidentally re-added. This does not make earlier public history private.

The active tree retains every experiment analysis, summary, validation CSV,
complete generation JSON, selected plots and artifact manifest; all scientific
source/config/tokenizer bytes, the app and final assignment evidence remain.
Internal receipts, duplicate derived comparisons, old execution plans, repeated
deployment screenshots and one-off report drafts are historical-only.
The verifier checks both archival ledgers and original Git blobs, with no
blanket missing-file exemptions. New internal documents belong in `outputs/`,
not among the submission files.

## Earlier cleanup history

The submission tree keeps the scientific implementation, every experiment's
analysis and canonical measurements, complete generations, final recovery
evidence, selected plots, application and final assignment evidence. Intermediate
plot exports, provisioning notebooks and development audits are available from
the immutable pre-cleanup public commit:

**`8f88edee89dfceb6d0a8c84e385a886c4e058356`**

[Browse the original complete repository](https://github.com/rayyan-merchant/Nano-Language-Model/tree/8f88edee89dfceb6d0a8c84e385a886c4e058356).
The earlier 1,103-file verification receipt remains unchanged
[in history](https://github.com/rayyan-merchant/Nano-Language-Model/blob/8f88edee89dfceb6d0a8c84e385a886c4e058356/docs/maintenance/verification_receipt.json).
It describes the old complete tree, not the new protection scope.

## Exact mapping

[public_tree.csv](public_tree.csv) classifies all 1,417 original paths with their
Git blob IDs, checkout sizes and disposition. `ARCHIVE` means removed from the
current tree, not erased from public history or made private. No external upload,
history rewrite, new public tag or private archive is required. For any listed
path, use `https://github.com/rayyan-merchant/Nano-Language-Model/blob/8f88edee89dfceb6d0a8c84e385a886c4e058356/PATH`.

To retrieve an exact file without replacing the current checkout:

```text
git show 8f88edee89dfceb6d0a8c84e385a886c4e058356:PATH
git rev-parse 8f88edee89dfceb6d0a8c84e385a886c4e058356:PATH
```

The latter must equal the recorded `git_blob`. Use a separate detached worktree
at that commit for a complete historical workflow, with its original dependency
profile and authorization constraints. Historical notebook links keep their
pinned commits. A shallow checkout must fetch history before historical-object
verification; do not silently skip it.

## Current verification

```text
python -m scripts.maintenance.verify_repository --output outputs/public-cleanup/verification.json
```

The new verifier checks the current retained protected artifacts independently
from the historical ledger. Scientific source/configuration/tokenizer paths and
bytes stay fixed. Research prose may only replace archived relative link targets
with pinned historical URLs; results, descriptions and generated strings remain
unchanged. All numerical research files and executed submission receipts remain
byte-protected, with Git text-newline normalization explicitly distinguished
from binary/package checksums. Missing historical objects are failures.

No new experiments, reserved-test evaluation, weight publication or automatic
merge is authorized by cleanup. Local generated outputs stay ignored.
