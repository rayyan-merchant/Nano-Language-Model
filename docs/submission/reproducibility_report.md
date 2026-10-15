# Reproducibility execution report — 2026-10-15

## Scope and status

**Linux clean-clone and actual fresh Colab CPU execution passed.** The original
research/selection/test results were not changed. The new submission PR24 remains
unmerged pending review. This is technical reproduction acceptance, not completion
of the academic report, comprehensive handout audit or recorded presentation.
The independent Windows clone also completed genuine download/loading/API/restart acceptance after the documented pip/temp/path corrections; its results are not inferred from Linux or the development environment.

Reviewed main: `aa4bb5f51f4760d533c50c290a0d42c60b911e89`.
Pinned clean source/notebook delivery: `43b93e14326efae76a2b106dfc493d150d73f3dd`.
Notebook runtime wrapper: `1f402d9e3b902e7d3e78565969476d2d7af06502` within that history.
The wrapper updates paths, headless plotting and receipts; no scientific modules,
frozen configs, tokenizers or accepted reports changed. Review status is separate
from code execution; do not describe these new wrapper commits as already reviewed.

## Independent environments and actual results

Path1 (colab-v3): fresh anonymous GitHub clone, isolated **Python3.11.14**,
**torch2.5.1+cpu**, **NumPy1.26.4**, **tokenizers0.21.0**, Linux6.6.122/glibc2.39,
CPU only, normal wheel installation and no global editable package. Native Colab
Python3.13.16/PyTorch2.11.0+cpu remained installed separately. Actual package paths
and complete dependency freeze are in [environment receipt](verification/colab-v3/environment/receipt.json).
No local D-drive packages, HF credentials, model cache, Drive mount or saved model
authentication were used. The initial clone was clean and every inference receipt
records `git_dirty=false`. Outputs and downloads use ignored fresh directories.

- Notebook core/numerical/retrieval checks: **23 passed, 1 dependency warning,4.36s**.
- Complete Python suite in that environment: **151 passed, 1 CUDA-only skip,
  1 dependency warning,46.41s**; actual log in [full-tests](verification/colab-v3/full-tests.txt).
- Existing `test_tiny_overfit` passed:60 synthetic-fixture updates and final loss
  asserted below15% of initial. No exact unlogged losses are invented.
- Genuine trained B01 step3000 attention:10 prompt tokens, all6 layers return
  `(1,6,10,10)` probabilities,32-dimensional heads, exactly zero upper triangle,
  earlier-logit max error **0.0** after future-suffix changes.
- Actual loaded parameters: E02 **2,173,056**, B01 **3,062,784**, F01 **6,046,560**.
  Strict safetensors state loading checks keys/shapes/finiteness/float32 and selected identities.
- Anonymous downloads: all6 exact file sizes and SHA256 values passed at immutable
  HF revision `719a7f3a4955225a9818e872b08c1503de601a81`.
- All3 models ×5 fixed prompts ×greedy/sampling =**30 genuine32-token cases**,
  temperature0.7, seed42. Raw token IDs, prompt IDs, complete unedited text and
  per-case timing are in [generation receipt](verification/colab-v3/generation/receipt.json).
- All **12 available canonical raw-token references matched exactly** on this
  platform/profile. The other18 cases have no prior export reference, explicitly
  labeled as such. This is not a general cross-platform bitwise guarantee.
- No replacement-character cases occurred among these30 short outputs. The
  tokenizer still uses its original replacement behavior for invalid generated
  UTF-8; historical research's malformed-byte limitations are not erased.
- In-process FastAPI readiness200 and B01 generation200 matched canonical decoding.
  Independent process restart reloaded all3 exact packages.
- First corrected fresh notebook: **13/13 code cells**, no errors, **106.352s**.

Path2 (colab-final): another **fresh CPU runtime**, independent /content, clone,
interpreter, packages and anonymous downloads. Standard nbclient0.10.2 with
nbformat5.10.4 executed the exact publicly fetched notebook top to bottom, retaining
its actual binary image output. **13/13 cells, zero errors, one embedded trained
attention image,120.689s measured notebook time**; the wrapping process took125.470s.
All30 generations,12/12 available raw-ID matches, trained causal invariance, API200
and process reload passed again. [Final receipts](verification/colab-final/receipts/summary.json)
and [unedited executed notebook](verification/colab-final/NanoLM_Submission_Demo_executed.ipynb).
Code and Markdown cells match the clean source exactly; nbformat validation passed.
This repeat did not reuse the first runtime's files, credentials or model cache.

The v3 connector omitted binary image MIME from its local notebook capture, even
though the plotting cell ran and created the actual PNG. We preserved that PNG
through verified export, then used standard Jupyter execution in the second fresh
runtime to save a complete executed notebook. No synthetic image or manually
invented cell output was inserted.

## Exact reproducible commands

[Submission README](README.md) gives the full pinned Windows setup and Linux
`.venv/bin/python` alternative. The Colab notebook explicitly installs uv0.12.7,
creates isolated Python3.11.14, upgrades pip26.2.1, installs torch2.5.1 from the CPU
wheel index, then requirements.txt/requirements-studio.txt and `pip install . --no-deps`.
`pip check` passed. Core commands actually executed in clean Linux environments:

```text
python -m pytest tests/test_core.py tests/test_submission.py -v
python -m pytest -q
python -m tools.submission_models --output model_artifacts
python -m tools.submission_demo --artifacts model_artifacts --section models --output outputs/submission-colab/models
python -m tools.submission_demo --artifacts model_artifacts --section attention --output outputs/submission-colab/attention
python -m tools.submission_demo --artifacts model_artifacts --section generation --output outputs/submission-colab/generation
python -m tools.submission_demo --artifacts model_artifacts --section api --output outputs/submission-colab/api
python -m tools.submission_demo --artifacts model_artifacts --section models --output outputs/submission-colab/restart
```

The helper CLI uses normal wheel imports for nano_lm; api/tools and immutable
research catalog records are loaded from the checkout root. A bare wheel without
that checkout is not advertised as a standalone distribution.

## Package identity and access

The [manifest](models.json) contains exact architecture, sizes, selected updates,
original research-checkpoint hashes, derived safetensors hashes, training commits,
canonical tokenizer checksum and research report paths. All three selected
packages are already-exported application artifacts, not newly trained checkpoints.
The owner authorized making the existing HF model repository public during this
phase; seven-file inventory was checked before the visibility change. The six
inference files plus Git attributes stayed at the identical revision. No full
optimizer/pickle checkpoint, secret or dataset was released. See [access](model_access.md).

## Genuine outputs and measured inference

Below are predetermined ROMEO sampling examples from the final fresh run, not
chosen by quality. All30 outputs, including greedy cases and poor samples, remain
in the receipt. Displayed text is unedited.

### E02 selected update5000

Sampling, temperature0.7, seed42,32 new tokens.

```text
ROMEO:
I cannot tell thee.

DUCHESS OF YORK:
Dear, by this, and I have forgive some speed
```

### B01 selected update3000

Sampling, temperature0.7, seed42,32 new tokens.

```text
ROMEO:
I cannot, my lord, and I know the hope,
Should not the rankful till I do:
Am I
```

### F01 selected update1750

Sampling, temperature0.7, seed42,32 new tokens.

```text
ROMEO:
I cannot tell thee.

DUCHESS OF YORK:
Therefore, I tell thee, and thou wilt be a king;

```

|Model|Historical validation BPB|Final fresh-Colab median32-token inference seconds|
|---|---:|---:|
|E02|2.02659521|0.161386|
|B01|1.98493095|0.316379|
|F01|1.98376817|0.504881|

These are mixed greedy/sampling short-inference observations, not training
runtimes, a robust benchmark or new validation scores. Single-seed BPB does not
establish generation coherence, significance or global architectural superiority.

## Evidence preservation and failures

First completed evidence ZIP:78,556bytes, SHA256
`a2f706bba455818c2e24f700ae43fc2a2e71315b83b6aae5ed3aa3df7034fcce`.
Final complete executed-notebook/receipt ZIP:159,180bytes, SHA256
`bdf8146fa1d85e6eab0a47e727ad475f24015a3984cb1236bf29f7c755996b04`.
Both were independently transferred, SHA256/ZIP-CRC/path checked, and extracted
as lightweight evidence. Neither contains weights. Both CPU runtimes were released
only after verified export; a later sessions query returned no active sessions.
Transfer receipts are [v3](verification/colab-v3-transfer.json) and
[final](verification/colab-final-transfer.json).

[Failure record](verification/failures.json): first real Colab attempt stopped at
external-environment module path reporting; second passed model loading but stopped
at inherited Matplotlib inline backend import. Both were wrapper issues and fixed
with focused regressions. The successful fresh executions occurred after those
corrections. The first Windows installer encountered DNS retries and an old-pip
metadata normalization failure; pinning pip26.2.1 corrects the documented setup.
A local harness retry also attempted checkout before clone completion; the failed
startup receipt was retained and rerun only after clone completion was verified.
No failed cell was hidden or relabeled successful.

## Acceptance boundaries

|Check|Observed status|
|---|---|
|Fresh clone/install/dependency health|PASS, two independent fresh Linux Colab environments|
|Scientific unit tests/tiny-overfit/causality|PASS;23 focused checks and151 full-suite passes|
|All3 public model downloads/identities/hashes|PASS, anonymous in both fresh environments|
|E02/B01/F01 genuine generation|PASS;30 cases each environment|
|Canonical raw-ID reference comparison|PASS for12 available cases each;18 have no prior reference|
|Actual trained attention|PASS; genuine PNG and embedded output visually checked|
|API and independent process reload|PASS|
|Actual Colab top-to-bottom execution|PASS twice,13 code cells each|
|Public notebook access|Public GitHub source fetched anonymously; corrected notebook rendered in Colab browser|
|Windows fresh setup/inference|PASS after environment corrections;149 full-suite passes plus2 corrected-path test passes,1 CUDA skip|
|GPU inference|NOT TESTED/NOT REQUIRED; CPU only|
|Protected science/config/tokenizers|1103 protected files unchanged; no scientific diff|
|Link/notebook audit|17 notebooks including new clean/executed copies; two pre-existing frozen link defects, zero new regressions|
|CI|Source43b93e1: all3 jobs passed, run38740618809; final evidence-head status checked separately|
|Human PR review/merge|PENDING; PR24, no automatic merge|
|Complete assignment/report/presentation|OUT OF SCOPE, next phase|

The handout's literal checkpoint-host examples mention GitHub/Drive. The user
chose approved public HF safetensors retrieval; whether the instructor requires
one of those literal hosts remains a question for the later full compliance audit,
not a fabricated handout acceptance claim. No original-test likelihood or new
research experiment was run. Only bounded trusted correctness fixtures optimize
synthetic models. Existing deployed service was checkedHTTP200; no new deployment
was performed, and existing hosting PR23 was not merged or rewritten by this phase.

## Independent Windows closure

[Windows summary](verification/windows/summary.json), [actual inference/token IDs](verification/windows/inference.json),
[anonymous download](verification/windows/download.json), [restart](verification/windows/restart.json),
[install command receipts](verification/windows/install-commands.json) and
[completion commands](verification/windows/completion-commands.json).
A separate GitHub clone at43b93e1 used a **new Python3.10.11 venv**, pinned
pip26.2.1/torch2.5.1+cpu and the documented requirements, with normal wheel install
and `pip check` passed. PYTHONPATH/PYTHONHOME/old virtual-env and HF token variables
were explicitly removed; Python user-site imports were disabled. This did not
use the old development venv, original D-drive checkpoints or cached model files.

Full suite after fixing temp-root permissions: **149 passed,1 skipped,2 storage
path failures** in136.89s. The two failures were WinError206 from the acceptance
clone's deeply nested Windows snapshot paths. Their unchanged assertions both
passed in a short unique writable temp directory (see [rerun log](verification/windows/storage-short-path-rerun.txt)).
Do not describe this as a single clean151-pass Windows invocation: all151 tests
were verified across the full run and targeted corrected-environment rerun.
Linux/CI separately passed all151 in one invocation. The earlier global Temp
ACL error and missing local output-parent harness error remain in the failure
record. No storage/model/scientific algorithm or OS security setting was changed.
Use a short Windows checkout/temp path for the complete storage suite. The
PowerShell submission commands create the output parent and use a unique temp;
ordinary core tests do not create the long research snapshot names.

All6 files were subsequently downloaded anonymously from the same pinned HF
revision, all3 actual selected models loaded,30 genuine generations completed,
all12 available raw-token references matched, trained future-logit error was0.0,
API/readiness returned200 and a fresh Python process reloaded all3 packages.
Actual new-venv environment/freeze, unedited text and timings remain in the receipt.

The public executed notebook link rendered the genuine summary/generations in
Colab; independent anonymous raw retrieval matched its exact149,462 exported bytes
and SHA256 `b8c3c257f7c90f7e62ff070d03a6ef6f3c64e8f9815c0ae0ab33a9bd5ff70f6c`.
[Notebook-access receipt](verification/notebook-access.json). No instructor Google
account was impersonated: public source access is verified, and each instructor
uses their own Google account for CPU execution. Human PR review remains pending.
