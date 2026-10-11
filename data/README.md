# Dataset provenance

nano_lm.prepare acquires the original Tiny Shakespeare corpus from
https://github.com/karpathy/char-rnn at revision
6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e, data/tinyshakespeare/input.txt.
Expected SHA256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
This is the original source behind the assignment's Tiny Shakespeare choice.
HTTP and pinned Git-object transport are available; TLS checks are never disabled.

Original UTF-8 bytes, source revision/URL/method/hash, contiguous 90/5/5 Unicode
boundaries, partition hashes and tokenizer training provenance are manifested.
Train-only character vocabulary is sorted with no special tokens or silent OOV.
BPE uses a fixed reversible 256-byte alphabet and training-only learned merges;
there is no text normalization or special-token insertion. Serialized artifacts
are reloaded, reconstructed and checked for byte conservation. Preparation refuses
to reuse an existing directory; failed partial artifacts remain for diagnosis.

Verified actual bytes: 1,115,394 total, 1,003,854 train, 55,770 validation, 55,770 test.
Actual vocab sizes: 65 character, 1024 BPE. Data copies remain ignored in Git.
Each training window comes from training token IDs only and cannot cross a split.
Overlapping windows within the same partition are allowed; this is not partition
leakage. Validation/test text never fits either tokenizer or drives training.
