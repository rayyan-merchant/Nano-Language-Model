# Data policy

Tiny Shakespeare is required. Intended source:
https://huggingface.co/datasets/karpathy/tiny_shakespeare
Download/preprocessing CLI is pending. Do not treat smoke fixtures as this dataset.
Record source revision, raw-file SHA256, original bytes, Unicode split boundaries
and decoded/re-encoded text hashes. Split 90/5/5 before fitting/tokenizing; no
windows cross boundaries. Fit character/BPE vocabulary only on train. Preserve
literal UTF-8 text and tokenizer artifacts; document unseen-token policies.
Raw/processed data are ignored. Do not commit large dataset copies.
