"""One real-data training engine for Windows, Kaggle and Colab."""
import argparse
import json
import os
import platform
import random
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from .artifacts import atomic_json, digest
from .bpe import encode_with_bytes, from_artifact, vocabulary_size
from .checkpoint import read_checkpoint, restore_training, save_checkpoint
from .config import load_experiment
from .evaluate import evaluate
from .model import GPT
from .prepare import validate


@dataclass(frozen=True)
class TrainSettings:
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    tokens_per_update: int = 4096
    micro_batch_size: int = 4
    gradient_clip: float = 1.0
    precision: str = "float32"
    device: str = "cpu"
    eval_every: int = 100
    checkpoint_every: int = 100
    conditioning_bytes: int = 256
    eval_max_bytes: int = 0
    eval_mode: str = "sliding"
    eval_batch_size: int = 32
    run_kind: str = "research"

    def validate(self, context):
        if self.seed not in {42, 123, 456}:
            raise ValueError("Study seeds are 42,123,456")
        if min(self.tokens_per_update, self.micro_batch_size, self.eval_every,
               self.checkpoint_every, self.conditioning_bytes, self.eval_batch_size) < 1:
            raise ValueError("Training sizes/intervals must be positive")
        if self.tokens_per_update % (context * self.micro_batch_size):
            raise ValueError("Effective token budget must divide by context * micro batch")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.gradient_clip <= 0 or self.eval_max_bytes < 0:
            raise ValueError("Invalid clipping or evaluation byte limit")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Invalid precision")
        if self.device not in {"cpu", "cuda"} or self.run_kind not in {"smoke", "benchmark", "research"}:
            raise ValueError("Invalid device/run kind")
        if self.eval_mode not in {"sliding", "blocks"}:
            raise ValueError("Invalid evaluation mode")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("Mixed precision is supported only on validated CUDA hardware")
        if self.device == "cuda" and self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16")


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def environment(root):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    source = {str(p.relative_to(root)): p.read_text(encoding="utf-8")
              for folder in ("src", "configs", "experiments/configs") for p in (root / folder).rglob("*")
              if p.suffix in {".py", ".json"}}
    freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return {"git_commit": commit.stdout.strip(), "git_dirty": bool(status.stdout.strip()),
            "code_hash": digest(source), "platform": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": freeze.stdout.splitlines(),
            "platform_label": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", os.environ.get("COLAB_RELEASE_TAG", "local"))}


def append_event(path, event):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def train(root, dataset, experiment_id, settings, steps, output_root, resume=None, model_override=None):
    root, dataset, output_root = Path(root), Path(dataset), Path(output_root)
    manifest = validate(dataset)
    variant = json.loads((root / f"experiments/configs/{experiment_id}.json").read_text())
    kind = variant["tokenizer"]
    tokenizer = from_artifact(json.loads((dataset / f"{kind}.json").read_text(encoding="utf-8")))
    config, defaults, variant = load_experiment(root, experiment_id, vocabulary_size(tokenizer))
    if model_override:
        if settings.run_kind != "smoke":
            raise ValueError("Model overrides are permitted only for labeled smoke checks")
        config = model_override
    settings.validate(config.context_length)
    if steps < 1 or (settings.run_kind != "research" and steps > 100):
        raise ValueError("Invalid training target; smoke/benchmark limited to 100 steps")
    if manifest["source"].get("kind") != "tiny_shakespeare" and settings.run_kind != "smoke":
        raise ValueError("Research requires pinned Tiny Shakespeare")
    env = environment(root)
    if settings.run_kind == "research" and (env["git_dirty"] or not env["git_commit"]):
        raise ValueError("Research requires a clean committed source checkout")
    seed_all(settings.seed)
    if settings.device == "cpu":
        torch.set_num_threads(1)
    model = GPT(config).to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=settings.precision == "float16")
    generator = torch.Generator().manual_seed(settings.seed)
    train_text = (dataset / "train.txt").read_bytes().decode("utf-8")
    validation_text = (dataset / "validation.txt").read_bytes().decode("utf-8")
    ids, lengths = encode_with_bytes(tokenizer, train_text)
    tokens = torch.tensor(ids)
    byte_offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    if len(ids) <= config.context_length:
        raise ValueError("Training partition shorter than shifted sequence")
    resolved = {"experiment_id": experiment_id, "model": config.to_dict(), "training": asdict(settings),
                "schedule": "constant", "tokenizer_kind": kind,
                "dataset_hash": manifest["manifest_hash"], "tokenizer_hash": tokenizer.checksum,
                "variant_hash": digest(variant), "code_hash": env["code_hash"]}
    config_hash = digest(resolved)
    state = {"step": 0, "training_tokens": 0, "training_bytes": 0, "train_seconds": 0.,
             "eval_seconds": 0., "best_loss": None, "best_checkpoint": None,
             "peak_gpu_bytes": 0, "amp_skipped_updates": 0, "history": [],
             "seen_bytes": np.zeros(len(train_text.encode()), dtype=np.bool_)}
    if resume:
        resume = Path(resume)
        payload, saved_tokenizer = read_checkpoint(resume)
        metadata = payload["metadata"]
        if metadata.get("schema_version") != 2 or metadata["config_hash"] != config_hash:
            raise ValueError("Resume dataset/tokenizer/config/code mismatch")
        if saved_tokenizer.checksum != tokenizer.checksum:
            raise ValueError("Resume tokenizer mismatch")
        directory = resume.parent.parent
        if json.loads((directory / "metadata.json").read_text()) != metadata:
            raise ValueError("Run metadata mismatch")
        checkpoints = list((directory / "checkpoints").glob("step_*.pt"))
        if resume.resolve() != max(checkpoints).resolve():
            raise ValueError("Resume latest committed checkpoint; do not fork/overwrite history")
        if metadata["environment"]["torch"] != env["torch"] or metadata["environment"]["gpu"] != env["gpu"]:
            raise ValueError("Exact resume requires matching PyTorch and accelerator")
        restore_training(payload, model, optimizer, scheduler, generator)
        scaler.load_state_dict(payload["scaler"] or {})
        state = payload["training_state"]
        append_event(directory / "events.jsonl", {"event": "resume", "step": state["step"], "environment": env})
    else:
        run_id = f"{experiment_id}_seed{settings.seed}_{uuid.uuid4().hex}"
        directory = output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {"schema_version": 2, "run_id": run_id, "resolved": resolved,
                    "config_hash": config_hash, "environment": env,
                    "parameters": model.parameter_count(), "source": manifest["source"],
                    "evaluation_text_hash": manifest["partitions"]["validation"]["sha256"]}
        atomic_json(directory / "metadata.json", metadata)
        atomic_json(directory / "tokenizer.json", tokenizer.artifact())
        atomic_json(directory / "dataset_manifest.json", manifest)
    if steps <= state["step"]:
        raise ValueError("Target total step must exceed checkpoint step")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[settings.precision]
    accumulation = settings.tokens_per_update // (config.context_length * settings.micro_batch_size)
    if settings.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lock = directory / ".active.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "host": platform.node()}))
    try:
        while state["step"] < steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if settings.device == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            loss_sum = 0.
            for _ in range(accumulation):
                starts = torch.randint(len(ids) - config.context_length, (settings.micro_batch_size,), generator=generator)
                x = torch.stack([tokens[s:s + config.context_length] for s in starts]).to(settings.device)
                y = torch.stack([tokens[s + 1:s + config.context_length + 1] for s in starts]).to(settings.device)
                for s in starts.tolist():
                    a, b = byte_offsets[s + 1], byte_offsets[s + config.context_length + 1]
                    state["training_bytes"] += int(b - a)
                    state["seen_bytes"][a:b] = True
                state["training_tokens"] += y.numel()
                with torch.autocast(settings.device, dtype=dtype, enabled=settings.precision != "float32"):
                    loss = model(x, y)[1]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss_sum += loss.detach().float().item() / accumulation
                scaler.scale(loss / accumulation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            previous_scale = scaler.get_scale()
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError("Nonfinite gradients")
            scaler.step(optimizer)
            scaler.update()
            if settings.device == "cuda":
                torch.cuda.synchronize()
            state["train_seconds"] += time.perf_counter() - began
            if scaler.get_scale() < previous_scale:
                state["amp_skipped_updates"] += 1
                if state["amp_skipped_updates"] > 20:
                    raise FloatingPointError("Repeated AMP overflow")
                continue
            scheduler.step()
            state["step"] += 1
            if settings.device == "cuda":
                state["peak_gpu_bytes"] = max(state["peak_gpu_bytes"], torch.cuda.max_memory_allocated())
            row = {"step": state["step"], "training_loss": loss_sum,
                   "training_tokens": state["training_tokens"], "training_bytes": state["training_bytes"],
                   "unique_target_bytes": int(state["seen_bytes"].sum()), "train_seconds": state["train_seconds"],
                   "tokens_per_second": state["training_tokens"] / state["train_seconds"],
                   "steps_per_second": state["step"] / state["train_seconds"],
                   "peak_gpu_bytes": state["peak_gpu_bytes"], "gradient_norm": float(norm),
                   "amp_skipped_updates": state["amp_skipped_updates"], "lr": scheduler.get_last_lr()[0],
                   "accumulation": accumulation}
            # Fixed cadence only: interruption stop points must not change model/RNG.
            if state["step"] % settings.eval_every == 0:
                began = time.perf_counter()
                val = evaluate(model, tokenizer, validation_text, settings.conditioning_bytes,
                               settings.eval_max_bytes, settings.eval_batch_size, settings.eval_mode)
                state["eval_seconds"] += time.perf_counter() - began
                row["validation"] = val
                row["generalization_gap"] = val["loss"] - loss_sum
            checkpoint = directory / "checkpoints" / f"step_{state['step']:06d}.pt"
            if "validation" in row and (state["best_loss"] is None or row["validation"]["loss"] < state["best_loss"]):
                state["best_loss"] = row["validation"]["loss"]
                state["best_checkpoint"] = checkpoint.name
            state["history"].append(row)
            append_event(directory / "events.jsonl", {"event": "step", "metric": row})
            if (state["step"] % settings.checkpoint_every == 0 or "validation" in row or state["step"] == steps):
                save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, generator,
                                state["step"], metadata, scaler=scaler, training_state=state)
        summary = {"schema_version": 2, "run_id": metadata["run_id"], "config_hash": config_hash,
                   "status": "completed_segment", "step": state["step"],
                   "checkpoint": checkpoint.name, "best_checkpoint": state["best_checkpoint"],
                   "metrics": state["history"][-1], "eval_seconds": state["eval_seconds"],
                   "history_hash": digest(state["history"])}
        atomic_json(directory / f"summary_{state['step']:06d}.json", summary)
        return directory, checkpoint, summary
    except BaseException as exc:
        append_event(directory / "events.jsonl", {"event": "failure", "step": state["step"],
                     "type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--experiment", default="E01")
    p.add_argument("--output", type=Path, default=Path("outputs/logs"))
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--resume", type=Path)
    for name, field in TrainSettings.__dataclass_fields__.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(p.parse_args())
    settings = TrainSettings(**{k: args.pop(k) for k in TrainSettings.__dataclass_fields__})
    directory, checkpoint, summary = train(args["root"], args["dataset"], args["experiment"], settings,
                                         args["steps"], args["output"], args["resume"])
    print(json.dumps({"run_dir": str(directory), "checkpoint": str(checkpoint), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
