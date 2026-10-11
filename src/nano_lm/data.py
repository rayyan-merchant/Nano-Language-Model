"""Contiguous split before tokenization; no cross-partition windows."""
import hashlib
import torch


def split_text(text: str):
    a, b = int(len(text) * 0.9), int(len(text) * 0.95)
    return {"train": text[:a], "validation": text[a:b], "test": text[b:]}, {
        "unit": "unicode_code_points", "boundaries": [0, a, b, len(text)],
        "sha256_utf8": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "policy": "contiguous 90/5/5; no normalization; tokenize each split separately"}


def sample_batch(tokens, context_length: int, batch_size: int, generator):
    if len(tokens) <= context_length:
        raise ValueError("Partition too short for shifted context")
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    x = torch.stack([tokens[s:s + context_length] for s in starts])
    y = torch.stack([tokens[s + 1:s + context_length + 1] for s in starts])
    return x, y
