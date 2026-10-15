# Submission reproducibility and Colab demo

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
git switch codex/submission-reproducibility-colab
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-studio.txt
.venv\Scripts\python -m pip install . --no-deps
.venv\Scripts\python -m pip check
.venv\Scripts\python -m pytest tests/test_core.py tests/test_submission.py -q
.venv\Scripts\python -m tools.submission_models --output model_artifacts
.venv\Scripts\python -m tools.submission_demo --artifacts model_artifacts --output outputs/submission-first
.venv\Scripts\python -m tools.submission_demo --artifacts model_artifacts --section models --output outputs/submission-restart
```

Linux uses `.venv/bin/python` for each command after `python -m venv .venv`.
The notebook provisions Python3.11.14 when the native Colab interpreter differs.
Run from the checkout root: the model catalog references immutable research records
in that checkout. Installing the Python wheel alone does not bundle those records.
No editable/global install, local D-drive paths, HF credential or model cache is used.
Use fresh output directories for subsequent invocations; reuse existing model
packages only with `--verify-only`, never by silently overwriting them.

This branch is reviewable submission implementation. Before final handoff, use the
exact accepted commit recorded in the execution report; a moving branch is for
review convenience, not an immutable research provenance identifier.
