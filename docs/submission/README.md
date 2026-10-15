# Current assignment audit

PR24 is merged. The [complete handout matrix](assignment_compliance_matrix.md), [ten required evidence items](evidence/README.md), and [readiness/remaining owner actions](assignment_readiness.md) are the current assignment-delivery index. Earlier PR-review statements below describe the reproducibility phase at capture time. The final Medium article and recording are separate pending deliveries.

# Submission reproducibility and Colab demo

**Verified:** independent fresh Linux/Colab and Windows clone inference, plus a complete executed Colab notebook. See the reports for observed environment corrections and remaining human review.

This phase verifies the selected E02 step5000, B01 step3000 and F01 step1750
inference packages. It does not retrain, select checkpoints or evaluate test data.
The final academic report, comprehensive assignment-evidence audit and recorded
presentation remain separate work.

- [Baseline](reproducibility_baseline.md)
- [Pinned models and provenance](models.json)
- [Public model retrieval](model_access.md)
- [Actual reproducibility report](reproducibility_report.md)
- [Colab instructions and actual status](colab_demo.md)
- [Clean submission notebook](../../notebooks/NanoLM_Submission_Demo.ipynb)

## CPU setup (Python3.10 or3.11)

```powershell
git clone https://github.com/rayyan-merchant/Nano-Language-Model.git
cd Nano-Language-Model
git checkout 43b93e14326efae76a2b106dfc493d150d73f3dd
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip==26.2.1
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-studio.txt
.venv\Scripts\python -m pip install . --no-deps
.venv\Scripts\python -m pip check
New-Item -ItemType Directory -Force outputs | Out-Null
$nanoTestTemp = Join-Path 'outputs' ('pytest-' + [guid]::NewGuid().ToString('N'))
.venv\Scripts\python -m pytest tests/test_core.py tests/test_submission.py -q --basetemp $nanoTestTemp
.venv\Scripts\python -m tools.submission_models --output model_artifacts
.venv\Scripts\python -m tools.submission_demo --artifacts model_artifacts --output outputs/submission-first
.venv\Scripts\python -m tools.submission_demo --artifacts model_artifacts --section models --output outputs/submission-restart
```

Linux uses `.venv/bin/python` for each Python command after `python -m venv .venv`; omit the PowerShell `$nanoTestTemp` line and use ordinary `python -m pytest tests/test_core.py tests/test_submission.py -q` (actually verified in Colab). The unique checkout-local Windows temp argument avoids an observed host global Temp ACL failure and preserves earlier test directories.
The notebook provisions Python3.11.14 when the native Colab interpreter differs.
Run from the checkout root: the model catalog references immutable research records
in that checkout. Installing the Python wheel alone does not bundle those records.
No editable/global install, local D-drive paths, HF credential or model cache is used.
Use fresh output directories for subsequent invocations; reuse existing model
packages only with `--verify-only`, never by silently overwriting them.

The checkout above pins the actual clean acceptance source. The submission PR
remains reviewable and unmerged; review status is distinct from technical execution.
The notebook pins wrapper source1f402d9 within that history; frozen scientific
files are byte-identical to reviewed mainaa4bb5f. No moving branch is used by the demo.

The immutable runtime checkout predates the final acceptance receipts by design. Use this current submission guide/report for closure evidence; the final documentation commits do not change its tested runtime implementation.
