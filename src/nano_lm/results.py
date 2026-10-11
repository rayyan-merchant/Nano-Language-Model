"""Validate and aggregate genuine completed research runs from either member."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from .artifacts import atomic_json, digest
from .checkpoint import read_checkpoint
from .config import load_experiment
from .bpe import from_artifact


def validate_run(directory, root, expected_steps=5000, allow_checks=False):
    directory, root = Path(directory), Path(root)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or digest(metadata["resolved"]) != metadata["config_hash"]:
        raise ValueError("Run schema/config hash mismatch")
    resolved = metadata["resolved"]
    cfg, _, variant = load_experiment(root, resolved["experiment_id"], resolved["model"]["vocab_size"])
    training = resolved["training"]
    if not allow_checks:
        if training["run_kind"] != "research" or resolved["model"] != cfg.to_dict():
            raise ValueError("Smoke/benchmark/modified model is not research evidence")
        if metadata["environment"]["git_dirty"] or not metadata["environment"]["git_commit"]:
            raise ValueError("Research code provenance incomplete")
    if digest(variant) != resolved["variant_hash"] or training["seed"] not in {42, 123, 456}:
        raise ValueError("Experiment definition/seed mismatch")
    summary_path = directory / f"summary_{expected_steps:06d}.json"
    if not summary_path.exists():
        raise ValueError(f"Missing complete {expected_steps}-step summary")
    summary = json.loads(summary_path.read_text())
    payload, tokenizer = read_checkpoint(directory / "checkpoints" / summary["checkpoint"])
    if payload["metadata"] != metadata or payload["step"] != expected_steps:
        raise ValueError("Checkpoint lineage/step mismatch")
    artifact = from_artifact(json.loads((directory / "tokenizer.json").read_text(encoding="utf-8")))
    if artifact.checksum != tokenizer.checksum:
        raise ValueError("Run tokenizer artifact mismatch")
    if tokenizer.checksum != resolved["tokenizer_hash"]:
        raise ValueError("Tokenizer mismatch")
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != resolved["dataset_hash"]:
        raise ValueError("Dataset provenance mismatch")
    history = payload["training_state"]["history"]
    if digest(history) != summary["history_hash"] or [r["step"] for r in history] != list(range(1, expected_steps + 1)):
        raise ValueError("Incomplete or altered committed history")
    if summary["config_hash"] != metadata["config_hash"] or summary["run_id"] != metadata["run_id"]:
        raise ValueError("Summary identity mismatch")
    last = history[-1]
    if last != summary["metrics"] or "validation" not in last:
        raise ValueError("Final validation missing or summary modified")
    evaluation = last["validation"]
    if evaluation["mode"] != training["eval_mode"]:
        raise ValueError("Evaluation policy mismatch")
    # Shared comparability key excludes seed and intended model/tokenizer factor,
    # retaining optimizer, precision, batching, evaluation, data and code policy.
    policy = {k: v for k, v in training.items() if k not in {"seed", "micro_batch_size"}}
    comparison = digest({"policy": policy, "dataset": resolved["dataset_hash"],
                         "code": resolved["code_hash"], "torch_version": metadata["environment"]["torch"],
                         "evaluation_text": metadata["evaluation_text_hash"]})
    return {"run_id": metadata["run_id"], "experiment_id": resolved["experiment_id"],
            "seed": training["seed"], "comparison_hash": comparison, "step": expected_steps,
            "validation_loss": evaluation["loss"], "validation_bpb": evaluation["bpb"],
            "scored_bytes": evaluation["scored_bytes"], "parameters": metadata["parameters"],
            "train_seconds": last["train_seconds"], "tokens_per_second": last["tokens_per_second"],
            "peak_gpu_bytes": last["peak_gpu_bytes"], "hardware": metadata["environment"]["gpu"],
            "tokenizer_kind": resolved["tokenizer_kind"], "history_hash": summary["history_hash"]}


def collect(directories, root, expected_steps=5000, allow_checks=False, require_complete=True):
    rows = [validate_run(p, root, expected_steps, allow_checks) for p in directories]
    if not rows or len({r["comparison_hash"] for r in rows}) != 1 or len({r["scored_bytes"] for r in rows}) != 1:
        raise ValueError("Incompatible data/code/training/evaluation policies or byte coverage")
    seen, unique = {}, []
    for row in rows:
        key = (row["experiment_id"], row["seed"])
        if key in seen:
            if row["run_id"] == seen[key]["run_id"] and row["history_hash"] == seen[key]["history_hash"]:
                continue  # Identical verified durable copies are deduplicated.
            raise ValueError("Duplicate experiment/seed; choose a prespecified run, do not cherry-pick")
        seen[key] = row
        unique.append(row)
    if require_complete and set(seen) != {(f"E{i:02d}", seed) for i in range(1, 15) for seed in (42, 123, 456)}:
        raise ValueError("Complete study requires all 14 configurations and three seeds")
    aggregate = []
    for eid in sorted({r["experiment_id"] for r in unique}):
        group = [r for r in unique if r["experiment_id"] == eid]
        item = {"experiment_id": eid, "n_seeds": len(group), "complete_seeds": len(group) == 3}
        for key in ("validation_bpb", "parameters", "train_seconds", "tokens_per_second", "peak_gpu_bytes"):
            values = [r[key] for r in group]
            item[key + "_mean"] = statistics.mean(values)
            item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(item)
    return unique, aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    rows, aggregate = collect(args.runs, args.root, args.steps, require_complete=not args.allow_incomplete)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "runs.json", rows)
    atomic_json(args.output / "aggregate.json", aggregate)
    for name, data in (("runs", rows), ("aggregate", aggregate)):
        with (args.output / f"{name}.csv").open("x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.errorbar([r["experiment_id"] for r in aggregate], [r["validation_bpb_mean"] for r in aggregate],
                yerr=[r["validation_bpb_sd"] or 0 for r in aggregate], fmt="o")
    ax.set(ylabel="Validation bits per source byte", title="Recorded means and sample SD; see seed completeness")
    fig.savefig(args.output / "validation_bpb.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for metric, label in [("tokens_per_second", "Training tokens per second"),
                          ("train_seconds", "Training seconds"), ("peak_gpu_bytes", "Peak GPU bytes")]:
        fig, ax = plt.subplots()
        ax.errorbar([r["experiment_id"] for r in aggregate], [r[metric + "_mean"] for r in aggregate],
                    yerr=[r[metric + "_sd"] or 0 for r in aggregate], fmt="o")
        ax.set(ylabel=label, title="Recorded means and sample SD")
        fig.savefig(args.output / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter([r["parameters_mean"] for r in aggregate], [r["validation_bpb_mean"] for r in aggregate])
    for r in aggregate:
        ax.annotate(r["experiment_id"], (r["parameters_mean"], r["validation_bpb_mean"]))
    ax.set(xlabel="Measured parameters", ylabel="Validation BPB", title="Parameter efficiency")
    fig.savefig(args.output / "parameter_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    for path in args.runs:
        summary = json.loads((path / f"summary_{args.steps:06d}.json").read_text())
        checkpoint, _ = read_checkpoint(path / "checkpoints" / summary["checkpoint"])
        history = checkpoint["training_state"]["history"]
        val = [r for r in history if "validation" in r]
        ax.plot([r["step"] for r in val], [r["validation"]["loss"] for r in val],
                label=checkpoint["metadata"]["run_id"][:24])
    ax.set(xlabel="Optimizer step", ylabel="Validation token loss (within tokenizer only)")
    if len(args.runs) <= 6:
        ax.legend(fontsize=6)
    fig.savefig(args.output / "validation_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
