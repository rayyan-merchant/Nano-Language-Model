"""End-to-end real-data verification, never a research/pilot run."""
import argparse
import json
from pathlib import Path
import torch
from .artifacts import atomic_json
from .checkpoint import read_checkpoint, load_for_inference
from .experiments import audit
from .generate import generate
from .prepare import validate
from .storage import export_run, restore_snapshot, verify_snapshot
from .train import TrainSettings, train


def equal_nested(a, b):
    import numpy as np
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def verify(root, dataset, output, device="cpu", precision="float32"):
    root, dataset, output = Path(root), Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = validate(dataset)
    if manifest["source"]["kind"] != "tiny_shakespeare":
        raise ValueError("End-to-end audit requires actual pinned Tiny Shakespeare")
    rows = audit(root, dataset)
    atomic_json(output / "configurations.json", rows)
    settings = TrainSettings(device=device, precision=precision, run_kind="smoke", tokens_per_update=128,
                             micro_batch_size=1, eval_every=2, checkpoint_every=1, eval_max_bytes=512)
    full, a, _ = train(root, dataset, "E01", settings, 4, output / "runs")
    partial, mid, _ = train(root, dataset, "E01", settings, 2, output / "runs")
    snapshot = export_run(partial, output / "exports")
    verify_snapshot(snapshot)
    restored = restore_snapshot(snapshot, output / "restored")
    _, b, summary = train(root, dataset, "E01", settings, 4, output / "restored",
                          resume=restored / "checkpoints" / mid.name)
    pa, _ = read_checkpoint(a)
    pb, _ = read_checkpoint(b)
    exact = {key: equal_nested(pa[key], pb[key]) for key in ("model", "optimizer", "scheduler", "scaler", "rng")}
    if not all(exact.values()):
        raise AssertionError(f"Resume diverged: {exact}")
    history_equal = all(x["training_loss"] == y["training_loss"] and x.get("validation") == y.get("validation")
                        for x, y in zip(pa["training_state"]["history"], pb["training_state"]["history"]))
    if not history_equal:
        raise AssertionError("Loss/validation progression diverged")
    bpe_dir, bpe_checkpoint, bpe_summary = train(root, dataset, "E02", settings, 2, output / "runs")
    samples = []
    for path in (b, bpe_checkpoint):
        model, tokenizer = load_for_inference(path)
        text = generate(model, torch.tensor([tokenizer.encode("To be")]), 8, greedy=True)
        samples.append({"checkpoint": str(path), "text": tokenizer.decode(text[0].tolist()),
                        "label": "2–4 step smoke output, not research-quality generation"})
    result = {"status": "integration_smoke_passed_not_experiment_go", "device": device,
              "precision": precision, "dataset": manifest, "configurations": rows,
              "exact_resume": exact, "history_equal": history_equal, "storage_restore": str(restored),
              "character_summary": summary, "bpe_summary": bpe_summary, "samples": samples,
              "limits": ["Validation restricted to 512 bytes for smoke; prefix 256 bytes excluded",
                         "Two independent Kaggle environments still require actual verification",
                         "No pilot or ablation results produced"]}
    atomic_json(output / "verification.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    args = p.parse_args()
    result = verify(args.root, args.dataset, args.output, args.device, args.precision)
    print(json.dumps({"status": result["status"], "exact_resume": result["exact_resume"],
                      "history_equal": result["history_equal"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
