# Public, immutable inference-package access

The owner requested public HF access on2026-10-15 after this phase initially
found private storage. The existing repository was inspected before its visibility
changed: six intended inference files plus Git attributes; no dataset, credentials
or full optimizer/pickle checkpoints. No file bytes or commit changed.

Repository: [rayyanmerchant/nanolm-studio-weights](https://huggingface.co/rayyanmerchant/nanolm-studio-weights)
Immutable revision: `719a7f3a4955225a9818e872b08c1503de601a81`.
The website's inference availability is independent of this download permission.
There is no requirement to authenticate, mount Drive or access the developer's PC.

| Model | Selected update | Inference bytes | Inference SHA256 |
|---|---:|---:|---|
| E02 |5000|8698368|5258e0db6839a2ee95b24083fc4bda43299c73d79310de5e8291d6bc3e4f7150|
| B01 |3000|12260216|0ea6ff2972740d2775f5d2da331d41395e2f75b36076436ec0b2a43ebcf18e3f|
| F01 |1750|24198360|b1e3d1c1f6c5386599b14e5311d59cc27d72743c6dfc5ea2dc9080fef8b09238|

The [submission manifest](models.json) records original research-checkpoint hashes
separately, exact model configurations, original training commits, report paths,
metadata hashes/sizes and the canonical tokenizer identity. These are derived
float32 safetensors packages, not new research checkpoints or training exports.

Download all six files from a clean checkout:

```text
python -m tools.submission_models --output model_artifacts
python -m tools.submission_models --output model_artifacts --verify-only
python -m tools.submission_demo --artifacts model_artifacts --output outputs/submission-demo
```

The helper builds explicit HTTPS resolve URLs using that full immutable revision.
It uses anonymous stdlib HTTP, no HF cache/token discovery, checks exact size and
SHA256, checks selected model metadata, and rejects partial/corrupt downloads.
Files are staged privately; the exclusively created destination cannot overwrite
an existing release. The complete download receipt is moved atomically last.
An interrupted publication may retain an incomplete destination without a receipt;
preserve it for diagnosis and retry to a new destination. No pickle deserialization
occurs. The canonical Registry additionally checks tensor keys/shapes/dtypes,
float32 finiteness, original identities and actual parameter counts on loading.

If HTTP403/404 occurs, stop and check repository access with its owner. Never
substitute another model or silently use random weights. No signed temporary URL
or private token should be placed in a notebook or receipt.
