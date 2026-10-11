"""Tiny CPU correctness run. Synthetic text is NOT a research experiment."""
import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .checkpoint import read_checkpoint, restore_training, save_checkpoint
from .config import ModelConfig
from .data import sample_batch
from .model import GPT
from .tokenizer import CharacterTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5, help="Total target steps, maximum 100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Smoke only: steps must be 1–100")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    text = "To be, or not to be.\n" * 30
    tokenizer = CharacterTokenizer.fit(text)
    config = ModelConfig(len(tokenizer.characters), d_model=24, n_heads=3,
                         n_layers=1, context_length=8, dropout=0.1)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Constant LR is an explicit smoke regime, independent of target stop step.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(args.seed)
    start = 0
    if args.resume:
        payload, saved_tokenizer = read_checkpoint(args.resume)
        if saved_tokenizer.checksum != tokenizer.checksum:
            raise ValueError("Smoke tokenizer mismatch")
        if args.resume.resolve().parent != args.run_dir.resolve():
            raise ValueError("Resume must preserve the original run directory")
        metadata = payload["metadata"]
        if metadata["seed"] != args.seed:
            raise ValueError("Resume seed mismatch")
        log = args.run_dir / "metrics.jsonl"
        history = [json.loads(line) for line in log.read_text().splitlines()]
        if not history or history[-1]["step"] != payload["step"]:
            raise ValueError("Log/checkpoint step mismatch; preserve and reconcile interrupted history")
        restore_training(payload, model, optimizer, scheduler, generator)
        start = payload["step"]
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        metadata = {"status": "synthetic_cpu_smoke", "seed": args.seed,
                    "git_commit": commit.stdout.strip() or "uncommitted",
                    "platform": platform.platform(), "torch": torch.__version__,
                    "device": "cpu", "precision": "float32", "parameters": model.parameter_count(),
                    "dataset": "synthetic smoke fixture", "tokenizer_hash": tokenizer.checksum,
                    "scheduler": "constant", "config": config.to_dict()}
        (args.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.steps <= start:
        raise ValueError("Resume target must exceed saved step")
    tokens = torch.tensor(tokenizer.encode(text))
    began = time.perf_counter()
    with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start + 1, args.steps + 1):
            x, y = sample_batch(tokens, 8, 2, generator)
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(x, y)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            row = {"step": step, "training_loss": loss.item(), "lr": scheduler.get_last_lr()[0],
                   "training_tokens": step * 16, "training_bytes": step * 16,
                   "gradient_norm": float(gradient_norm),
                   "segment_elapsed_seconds": time.perf_counter() - began}
            log.write(json.dumps(row) + "\n")
            log.flush()
    path = args.run_dir / f"step_{args.steps:06d}.pt"
    save_checkpoint(path, model, tokenizer, optimizer, scheduler, generator, args.steps, metadata)
    print(json.dumps({"status": "synthetic_cpu_smoke", "steps": args.steps,
                      "checkpoint": str(path), "last_loss": loss.item()}))


if __name__ == "__main__":
    main()
