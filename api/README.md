# NanoLM Studio API

Thin FastAPI serving wrappers around the frozen nano_lm package. Selected identities
are anchored to accepted research manifests: E02 5000, B01 3000 and F01 1750.
Serving loads verified float32 safetensors only; trusted offline pickle conversion
is confined to tools/package_models.py. There are no checkpoint uploads or external
LLM requests. Read [operation/deployment guide](../docs/deployment/README.md) and
[actual verification](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/docs/deployment/verification.md) before claiming acceptance.

Start from repository root with `python -m uvicorn api.main:app --host 127.0.0.1
--port 8000 --workers 1`. Set NANOLM_ARTIFACTS to an immutable private package root.
Liveness is separate from all-model readiness. One inference worker, finite request
limits and canonical per-call sampling generators prevent silent model substitution
and overlapping model mode changes. Error clients receive no internal paths.
