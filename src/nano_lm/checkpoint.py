"""Versioned trusted local checkpoints, with complete CPU smoke RNG state."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from .config import ModelConfig
from .model import GPT
from .bpe import from_artifact, vocabulary_size


def save_checkpoint(path: Path, model, tokenizer, optimizer, scheduler, generator, step, metadata, scaler=None, training_state=None):
    if path.exists():
        raise FileExistsError(path)
    payload = {"format_version": 1, "config": model.config.to_dict(),
               "model": model.state_dict(), "tokenizer": tokenizer.artifact(),
               "tokenizer_hash": tokenizer.checksum, "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None, "step": step,
               "training_state": training_state,
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(), "sampling": generator.get_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive temporary file prevents concurrent writers using the same path.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # Atomic create-if-absent on both Windows and Linux; never replace an existing target.
    os.link(temporary, path)
    temporary.unlink()


def read_checkpoint(path):
    # Only load our trusted artifacts: pickle-based optimizer/RNG data are not untrusted input.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = from_artifact(payload["tokenizer"])
    if tokenizer.checksum != payload["tokenizer_hash"]:
        raise ValueError("Tokenizer checksum mismatch")
    if payload["config"]["vocab_size"] != vocabulary_size(tokenizer):
        raise ValueError("Tokenizer/model vocabulary mismatch")
    return payload, tokenizer


def load_for_inference(path):
    payload, tokenizer = read_checkpoint(path)
    model = GPT(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer


def restore_training(payload, model, optimizer, scheduler, generator):
    if payload["config"] != model.config.to_dict():
        raise ValueError("Checkpoint/model configuration mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    generator.set_state(payload["rng"]["sampling"])
    if payload["rng"]["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly restore CUDA RNG on CPU")
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
