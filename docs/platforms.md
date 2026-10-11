# Kaggle and Colab environments

Use the same pinned code commit and train/evaluate entry points on both accounts.
No training code is duplicated in notebooks. Dataset/run/export paths are CLI
arguments. CUDA must be explicitly requested; absence is a hard failure. Baseline
precision is float32. float16 uses torch.autocast plus CUDA GradScaler, unscales
before clipping, detects skipped optimizer updates and advances the scheduler
only on actual optimizer updates. bfloat16 requires hardware support and does not
need scaling. All CUDA RNG and sampling state are checkpointed.

Locked research dependency profile is Python 3.10/3.11, torch 2.5.1, numpy 1.26.4.
Provision an isolated environment if the notebook's current Python is incompatible.
Modern Colab Python 3.13 cannot use this older lock directly. Do not silently
upgrade requirements or overwrite a GPU runtime with CPU PyTorch. A separately
qualified newer profile requires documented approval before the main study.

This task's actual Colab T4 runs used Python 3.13.15 and torch 2.11.0+cu130.
Those smoke results prove execution/AMP resume on that environment, not equivalence
to the locked environment. No independent Kaggle account was tested in this task.
Record GPU, framework/CUDA/package versions, OS, precision and hashes per run.
No bitwise equivalence across different GPUs/frameworks is promised. The collector
requires matching code, framework, effective-token and evaluation policies and
records different GPU hardware as a confound; re-profile rather than assume speed.

Cloud research requires an export root, but users must verify its durability.
Save/export and restore before stopping a runtime. Respect each member's legitimate
account quotas; no account switching, quota circumvention or paid service needed.

## Rayyan's tested Kaggle profile (2026-10-11)

See docs/qualifications/KAGGLE_RAYYAN_20261011.md. The legitimate Rayyan
account passed on Tesla T4 with isolated Python3.11.14, torch2.5.1+cu124,
CUDA12.4 and NumPy1.26.4. Native Python3.13.15/torch2.11 was left untouched.
The qualification builder provisions the approved primary pins through uv,
checks CUDA and pip compatibility, and records a resolved freeze; compare all
transitive versions before accepting another account. It invokes shared package
entry points rather than duplicating notebook training. Saved output mounted
in a fresh Kaggle notebook passed generation and exact AMP continuation.
This satisfies one account's gate; the other member must independently repeat
the profile and storage test before study approval. No policy change is implied.
