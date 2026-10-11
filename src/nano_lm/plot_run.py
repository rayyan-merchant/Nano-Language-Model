"""Regenerate labeled loss/throughput figures from committed checkpoint history."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from .checkpoint import read_checkpoint


def plot_checkpoint(checkpoint, output):
    payload, _ = read_checkpoint(checkpoint)
    state = payload.get("training_state")
    if not state or not state["history"]:
        raise ValueError("Checkpoint has no committed research-engine metric history")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = state["history"]
    label = payload["metadata"]["resolved"]["training"]["run_kind"]
    fig, ax = plt.subplots()
    ax.plot([r["step"] for r in history], [r["training_loss"] for r in history], label="training")
    val = [r for r in history if "validation" in r]
    ax.plot([r["step"] for r in val], [r["validation"]["loss"] for r in val], label="validation")
    ax.set(xlabel="Optimizer step", ylabel="Token loss", title=f"{label}: recorded losses (not cross-tokenizer comparable)")
    ax.legend()
    fig.savefig(output / "loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r["step"] for r in history], [r["tokens_per_second"] for r in history])
    ax.set(xlabel="Optimizer step", ylabel="Cumulative training tokens/sec", title=f"{label}: measured training throughput")
    fig.savefig(output / "throughput.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(plot_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
