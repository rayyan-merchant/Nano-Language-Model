# Assignment readiness and delivery status

Audit date: 2026-10-15. Starting reviewed main:
`ee2b941311cfb451a73dbc65207ad3dfabb5977f`. PR24 was merged before this branch
was created. The [complete matrix](assignment_compliance_matrix.md) maps every
handout section and rubric category to content, execution evidence and actions.
The [ten-figure index](evidence/README.md) provides captions and working links.

**Status: technical evidence and article publication verified; overall submission
acceptance remains unconfirmed pending delivery evidence and instructor access
acceptance.** The original technical audit did not itself produce the article,
recording or presentation; their updated statuses are recorded below.

## Delivery refresh — 2026-10-17

The [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca) is publicly readable and attributed to Rayyan Merchant;
its ten main sections were inspected in the browser. Automated HTTP access returned
403, while the browser successfully displayed the article without sign-in.
The owner reports that the demo recording and class presentation have both been
delivered. Their links/receipts have not yet been supplied, so duration, content
and delivery are not independently verified. See the
[delivery receipt](verification/delivery_status_20261017.json).

The October 15 technical audit below remains dated evidence. Publication is now
verified; course submission and instructor acceptance are separate from publication.

## Readiness by delivery

| Area | State | Evidence / remaining action |
|---|---|---|
| From-scratch scientific implementation and accepted research | PASS, preserved | Explicit Q/K/V, masking, decoder, genuine E01–E14 and combined F01/B01 records; no new scientific change |
| All ten Section12 evidence items | PASS with disclosed caveats | [Index](evidence/README.md); eight new numbered diagrams, two existing figures reused without copying; original hand work plus corrected typed companion |
| Handwritten attention | PASS with rounding correction | [Audit](evidence/hand_calculation_audit.md); original4 pages preserved. Small rounding/truncation inconsistencies corrected, not a model bug |
| Seven concepts and MLM/CLM comparison | PASS for technical material | [Concepts](evidence/concepts_and_objectives.md), exact educational masks and numerical comparison; no trained BERT claim |
| Selected model inference/reload | PASS on executed CPU receipts | E02 step5000; B01 step3000; F01 step1750. Actual counts/config/hash identities checked in both Colab runs and fresh-process receipts |
| Clean-environment acceptance | PASS within documented platforms | Linux151 passed/1 CUDA skip; Windows149 +2 corrected-path passes; two independent Colab CPU executions. One final executed notebook retained, earlier execution represented by receipts/transfer record |
| Public pinned model packages | PASS for technical anonymous access | [Fresh access receipt](verification/assignment_public_access.json); full tensor bytes independently verified in prior fresh environments |
| Literal GitHub/Drive checkpoint route | PARTIAL — instructor decision | HF is publicly retrievable but is not literally the handout's GitHub/shareable Drive route. Ask instructor if acceptable. No mirror is authorized or created |
| Deployment | Implemented, time-dependent | [Deployment records](../deployment/README.md); existing receipts establish prior live operation. This audit's Figure10 proves actual local API execution, not present public uptime |
| Generation quality | PARTIAL | Genuine samples include coherent local fragments, repeated phrases, malformed words and incomplete endings; no guaranteed coherence grade or significance |
| Historical source attribution and individual understanding | Human review pending | New equations/figures attributed. Each author must review historical adaptations, confirm contributions and explain implementation orally |
| Final integrated Medium article | PUBLICATION VERIFIED | [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca); public title/author and section coverage checked October 17. Publication does not prove course submission or instructor approval of length/content |
| Screen-recorded demo | OWNER-REPORTED DELIVERED; EVIDENCE PENDING | Owner confirmed delivery October 17; provide the recording link. Duration (5–7 minutes), walkthrough content and access have not been checked |
| Class presentation / oral defense | OWNER-REPORTED DELIVERED; EVIDENCE PENDING | Owner confirmed delivery October 17; provide a delivery receipt or instructor acknowledgement. No independent attendance/oral assessment claimed |
| Final review and submission | PENDING HUMAN DELIVERY | Retain evidence of course submission, provide the recording/presentation evidence, and resolve the HF access-acceptance question |

## Historical technical acceptance evidence — October 15

The [prior full Linux log](verification/colab-v3/full-tests.txt) contains
`151 passed, 1 skipped` in46.41s. The CUDA-only skip is expected for that CPU
environment, not a failed or executed GPU test. The current phase did not allocate
GPU/Colab or rerun research. The test source's tiny-overfit assertion checks a
trusted synthetic60-update fixture for final loss below15% of initial; its passing
execution is preserved. Synthetic loss is never labeled Shakespeare evidence.

Both [earlier Colab summary](verification/colab-v3/summary.json) and
[final summary](verification/colab-final/receipts/summary.json) report30 genuine
cases and12/12 available raw-ID reference matches, with no research updates.
All six package file sizes/hashes are compared to the immutable manifest; actual
model architecture/parameter/selected-step identities agree across model and
fresh-process receipts. The final notebook's13 executed code cells contain no
error outputs and retain the genuine trained attention PNG. The earlier
[transfer receipt](verification/colab-v3-transfer.json) records13 cells/no errors;
only the final executed notebook is retained here. Neither is an instructor
account test or a recorded presentation.

The saved API's HTTP200 result matches the canonical B01 generation byte-for-byte
and validates against the current request/response schemas. B01's10×10 saved
attention matrix has normalized rows, an exactly zero upper triangle and
future-token logit error0.0. Layer0/head0 and selected step3000 are explicit.
F01's40 validation rows match the250-step cadence through10000; its minimum is
step1750 and agrees with the summary/figure. Every E01–E14 parameter count,
completed-update count and best BPB in the presentation CSV is checked against its
individual accepted summary. Full original training/recovery is not re-executed.

## Access and checkpoint interpretation

[Model instructions](model_access.md) and [models.json](models.json) include
all paths, byte sizes, SHA256, original research checkpoint hashes and selected
configurations. Public HF revision is
`719a7f3a4955225a9818e872b08c1503de601a81`. The fresh anonymous API query confirms
that exact revision is public, all six file sizes agree, tensor LFS SHA256 values
match, and all three small metadata files freshly download with the expected
hashes. Large weights were not redundantly downloaded during this audit; the
prior two independent Colab and Windows downloads/reloads provide that evidence.
Inference-only safetensors are deliberately distinct from full resumable research
checkpoints. E02's selected5000 is its terminal state; B01/F01 use validation-best
submission states, not their terminal4500/10000 states.

**Question for instructor:** “May the public version-pinned Hugging Face
inference checkpoints and verified Colab downloader satisfy the checkpoint-access
requirement instead of GitHub or shareable Google Drive?” If not, the owner must
authorize a mirror, then publish the same verified bytes, record checksums and
test anonymous instructor retrieval. This audit does not publish another copy.

## Remaining delivery verification

The [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca) now covers the main report themes: problem,
foundations, implementation, data/scoring, experimental setup, results, ablations,
failure modes, inference/reproducibility and conclusion. The compliance matrix
records the observed coverage; publication is not an instructor grading decision.
Author attribution/format review and evidence of course submission remain distinct.

The owner reports recording and presentation delivery. Remaining maintenance is to
attach their actual links/receipts and verify access, duration and the Section 17
walkthrough items. Do not describe them as not yet produced, and do not infer a
verified recording from the executed Colab notebook. No new training is required.

## Historical audit commands and verification — October 15

Executed from this checkout using existing Windows Python3.10.11 CPU environment:

```text
git fetch origin
git switch -c codex/assignment-evidence-audit origin/main
pdftoppm -png -r 125 docs/handwritten-assignment.pdf outputs/assignment-audit/handwritten
python -m tools.assignment_evidence
python -m pytest tests/test_assignment_evidence.py tests/test_core.py tests/test_submission.py -q --basetemp .pytest_cache/audit15
python -m scripts.maintenance.verify_repository --output outputs/assignment-audit/integrity.json
```

The new focused invocation passed **28 tests in46.79s**, with15 dependency
deprecation warnings (Matplotlib/pyparsing and Starlette/httpx), not scientific
failures. It includes the ordinary bounded tiny fixture, not new substantive
training. The builder initially required two retained executed notebooks; inspection
showed only the final notebook is kept, so the audit now correctly verifies the
earlier execution through its receipts and transfer record. That initial audit
assertion failure did not reflect a failed Colab run. No original receipt changed.

Original11-page handout read/hash verified, original4-page handwritten work
rendered/inspected, all generated/reused figures visually reviewed, source input
hashes recorded, public access refreshed, and notebook outputs/relative links
checked. The final integrity/CI outcome is recorded in the accompanying
[audit execution receipt](verification/assignment_checks.json).

The final six audit-specific tests passed again in1.21s after the hash portability
correction. A committed-byte check found existing Windows CRLF checkout differences
in several source JSON/CSV files. The audit now explicitly records LF-normalized
text-source hashes alongside the observed Windows raw-byte hashes. Exact binary,
model-weight and inference-package checksums are never normalized. This changes
only new audit hashing; original scientific files and receipts remain untouched.
Final branch CI is tracked on
[PR25](https://github.com/rayyan-merchant/Nano-Language-Model/pull/25).

No model/config/tokenizer bytes, accepted reports, selected identities or immutable
verification receipts are changed. No new research updates, original-test
assessment, new model package, mirror publication or presentation was performed.
