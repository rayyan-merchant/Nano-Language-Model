"""Configuration audit, short benchmarking and single-run execution."""
import argparse
import json
from pathlib import Path
import torch
from .artifacts import atomic_json
from .bpe import from_artifact
from .config import load_experiment
from .model import GPT
from .prepare import validate
from .train import TrainSettings, train


def audit(root, dataset):
    manifest = validate(dataset)
    torch.set_num_threads(1)
    base, _, _ = load_experiment(root, "E01", manifest["tokenizers"]["character"]["vocab_size"])
    expected = {"E01": {}, "E02": {}, "E03": {"position": "rope"}, "E04": {"position": "learned"},
        "E05": {"activation": "relu"}, "E06": {"n_heads": 4}, "E07": {"n_heads": 8},
        "E08": {"n_layers": 6}, "E09": {"n_layers": 8}, "E10": {"mlp_expansion": 2},
        "E11": {"context_length": 64}, "E12": {"context_length": 256},
        "E13": {"d_model": 144}, "E14": {"d_model": 240}}
    rows = []
    for eid, intended in expected.items():
        variant = json.loads((root / f"experiments/configs/{eid}.json").read_text())
        vocab = manifest["tokenizers"][variant["tokenizer"]]["vocab_size"]
        cfg, _, _ = load_experiment(root, eid, vocab)
        delta = {k: v for k, v in cfg.to_dict().items() if k != "vocab_size" and base.to_dict()[k] != v}
        if delta != intended or variant["tokenizer"] != ("bpe" if eid == "E02" else "character"):
            raise ValueError(f"Unapproved configuration change in {eid}")
        model = GPT(cfg).eval()
        with torch.no_grad():
            logits, _, heads = model(torch.zeros((1, cfg.context_length), dtype=torch.long), return_attention=True)
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite {eid} outputs")
        rows.append({"experiment_id": eid, "parameters": model.parameter_count(),
                     "logits": list(logits.shape), "head_shape": list(heads[0].shape),
                     "intended_delta": delta, "vocab_size": vocab})
        del model, logits, heads
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--benchmark-all", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--experiment", default="E01")
    args = p.parse_args()
    if args.audit:
        atomic_json(args.output, audit(args.root, args.dataset))
    elif args.benchmark_all:
        if args.steps > 100:
            p.error("Benchmark limited to 100 steps")
        results = []
        for i in range(1, 15):
            directory, checkpoint, summary = train(args.root, args.dataset, f"E{i:02d}",
                TrainSettings(device=args.device, run_kind="benchmark"), args.steps, args.output)
            row = summary["metrics"]
            results.append({"experiment_id": f"E{i:02d}", "run_dir": str(directory),
                "steps_per_second": row["steps_per_second"], "tokens_per_second": row["tokens_per_second"],
                "peak_gpu_bytes": row["peak_gpu_bytes"],
                "projected_5000_train_seconds": 5000 / row["steps_per_second"],
                "projection_limit": "training only; excludes validation/checkpoint IO; warmup included"})
        atomic_json(args.output / "benchmark.json", results)
    else:
        raise ValueError("Use nano_lm.train for one config; batch main-study execution is intentionally explicit")


if __name__ == "__main__":
    main()
