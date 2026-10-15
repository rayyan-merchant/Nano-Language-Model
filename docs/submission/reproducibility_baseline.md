# Reproducibility baseline — 2026-10-15

Reviewed main: `aa4bb5f51f4760d533c50c290a0d42c60b911e89`; initially clean.
Work starts from that commit on `codex/submission-reproducibility-colab`.
Hosting PR23 is separate and unmerged; this branch does not import its changes.

Baseline tests in the existing development environment: **140 passed, 1 CUDA-only
skip, 15 dependency warnings, 162.76 seconds**. This is a baseline, not clean-clone
acceptance. Research pins: Python3.10/3.11, torch2.5.1, NumPy1.26.4,
tokenizers0.21.0. Frontend Node24/Next16.4.1; Node is unnecessary for Python/Colab.
Existing notebooks:15, all historical paths preserved. Current API Registry loads
only hash-verified safetensors, strict model metadata and canonical tokenizer;
canonical nano_lm.generate supplies inference. tools.package_models already exported
the three selected original states; no re-export is planned.

Existing packages: E02 selected5000, B01 selected3000, F01 selected1750, available
locally and in immutable HF revision719a7f3a4955225a9818e872b08c1503de601a81.
Original and derived hashes remain separate in models.json. Initially private;
owner subsequently requested public HF access. Visibility changed only after
inventory verification: six inference files plus Git attributes, no research
pickle checkpoints, datasets or credentials. Anonymous metadata access passed.
Full anonymous fresh download remains an independent acceptance gate.

Website and /api/ready returnedHTTP200 on live checks this task. Earlier deployment
receipts alone were not used to assert current accessibility. Hosting source and
build evidence remain in separate PR23. No new deployment is authorized here.

A fresh CPU Colab allocation is available (no GPU). Native environment observed
Python3.13.16, torch2.11.0+cpu, numpy2.1.3, tokenizers0.23.2. These do not match the
supported package pins, so the notebook will provision isolated Python3.11.14 and
CPU torch2.5.1 without replacing native packages. Research configuration/source
will remain unchanged. Colab execution and shareable notebook accessibility must
be verified separately; connector runtime URL is not an instructor notebook link.

Original 11-page A1_AIE.pdf is available read-only at the supplied local path.
The comprehensive handout audit/report/presentation remains a later phase.
