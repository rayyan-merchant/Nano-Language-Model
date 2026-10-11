"""Strict model configuration and locked experiment resolution."""
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    context_length: int = 128
    mlp_expansion: int = 4
    activation: str = "gelu"
    position: str = "sinusoidal"
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.n_heads, self.n_layers,
               self.context_length, self.mlp_expansion) < 1:
            raise ValueError("Dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.position == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.position not in {"sinusoidal", "learned", "rope"}:
            raise ValueError("Unsupported position method")
        if self.activation not in {"gelu", "relu"} or not 0 <= self.dropout < 1:
            raise ValueError("Invalid activation or dropout")

    def to_dict(self):
        return asdict(self)


def load_experiment(root: Path, experiment_id: str, vocab_size: int):
    base = json.loads((root / "configs/baseline.json").read_text())
    variant = json.loads((root / f"experiments/configs/{experiment_id}.json").read_text())
    model = ModelConfig(vocab_size=vocab_size, **base["model"])
    model = replace(model, **variant["model_overrides"])
    return model, {**base["training"]}, variant
