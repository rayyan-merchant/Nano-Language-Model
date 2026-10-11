# Independent Kaggle GPU qualification: `srijaali`

**Status: NO-GO.** This is development qualification evidence, not a research run.
No 5,000-step pilot or ablation experiment was started.

## Provenance and account

- Local clean `main` after the merged PR: `b2ef1ca842ff6b0794b802f1baae58226732674f`.
- Official Kaggle CLI: 2.2.4, in an existing isolated local environment.
- Authenticated Kaggle account: `srijaali` (username field only; credentials were not displayed).
- Official quota query before and after probes: 30.00 GPU hours remaining, 0.00 hours used; 20.00 TPU hours remaining.
- Private notebook: <https://www.kaggle.com/code/srijaali/nlm-qualification-environment-probe-2026-10-11>.
- Kaggle CLI metadata pull for version 1 confirmed `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, and `is_private: true`.

## GPU provisioning probes

| Version | Request | Result |
| --- | --- | --- |
| 1 | `enable_gpu: true` in private notebook metadata | Notebook ran on CPU and deliberately failed the CUDA requirement. |
| 2 | Explicit `NvidiaTeslaT4` in metadata and `kaggle kernels push --accelerator NvidiaTeslaT4` | Same CPU-only result. |
| 3 | Explicit `NvidiaL4` in metadata and `kaggle kernels push --accelerator NvidiaL4` | Same CPU-only result. |

All three notebook executions reported Python 3.13.15 on Linux x86_64, PyTorch
2.11.0+cpu, `torch.version.cuda` null, `torch.cuda.is_available()` false,
no `/dev/nvidia0`, and no `nvidia-smi`. Installed package versions reported by
the probe were NumPy 2.1.3, tokenizers 0.23.1, pytest 8.4.2, and matplotlib
3.10.0. GPU model and memory could not be measured because no GPU was attached.
The notebook intentionally raised `RuntimeError` when CUDA was unavailable; the
three `KernelWorkerStatus.ERROR` results are expected from that guard. No package
was installed or replaced in Kaggle.

Kaggle's official CLI documentation identifies `NvidiaTeslaT4` as the GPU T4 x2
request and accepts `--accelerator` for pushes:
<https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md>.
The server accepted the requests, but did not provide GPU hardware. The cause
is not yet established; an account-side or platform-side setting requires
inspection in Kaggle's browser UI or support.

## Qualification checks not completed

- Pinned research profile (Python 3.10/3.11, torch 2.5.1, NumPy 1.26.4): not
  available in the observed default runtime and not installed. No policy change
  is proposed without a separately tested, approved shared GPU profile.
- Tiny Shakespeare acquisition and character/BPE/manifest hashes: **not
  independently checked inside Kaggle**. The audited reference hashes remain
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
  `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`,
  `2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55`,
  and `76b594a1979107795734f8d2cda08b56c1520c8d44883dbaa5a61dbee53ba16f`.
- Kaggle test suite, CUDA/AMP real-data verification, all-14 short benchmark,
  longer representative benchmark, full-validation timing, export timing, and
  durable artifact recovery: **not run**. No checkpoint was produced on Kaggle.
- GPU quota was unchanged at 30.00 hours after the failed probes.

## Required next steps

1. Confirm this account's GPU eligibility and notebook accelerator settings in
   Kaggle's UI, or ask Kaggle support why accepted GPU notebook versions launch
   without a GPU. Re-run a private environment probe and require an NVIDIA
   device, a CUDA PyTorch build, and `torch.cuda.is_available() == True` before
   installing or training anything.
2. Qualify the original pinned GPU profile in an isolated Kaggle environment;
   if infeasible, test one compatible profile on both members' accounts and seek
   approval before changing research policy.
3. Only then run the documented Kaggle tests, real-data CUDA verification,
   benchmarks, full-validation timing, and actual output-version recovery gate.

The research readiness decision remains **NO-GO**.
