import math
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from nano_lm.attention import merge_heads, scaled_attention, split_heads
from nano_lm.config import ModelConfig, load_experiment
from nano_lm.data import sample_batch, split_text
from nano_lm.generate import generate
from nano_lm.model import GPT, TransformerBlock
from nano_lm.positions import apply_rope, sinusoidal
from nano_lm.tokenizer import CharacterTokenizer


def tiny(**overrides):
    return replace(ModelConfig(9, d_model=24, n_heads=3, n_layers=1,
                               context_length=8, dropout=0), **overrides)


def test_tokenizer_roundtrip(tmp_path):
    text = "é🙂abc\n"
    tokenizer = CharacterTokenizer.fit(text)
    assert tokenizer.checksum == CharacterTokenizer.fit(text[::-1]).checksum
    assert tokenizer.decode(tokenizer.encode(text)) == text
    tokenizer.save(tmp_path / "tokenizer.json")
    assert CharacterTokenizer.load(tmp_path / "tokenizer.json").checksum == tokenizer.checksum
    with pytest.raises(ValueError):
        tokenizer.encode("z")


def test_split_and_shift():
    text = "a" * 90 + "b" * 5 + "c" * 5
    parts, metadata = split_text(text)
    assert parts == {"train": "a" * 90, "validation": "b" * 5, "test": "c" * 5}
    assert metadata["boundaries"] == [0, 90, 95, 100]
    tokens = torch.arange(20)
    x, y = sample_batch(tokens, 4, 5, torch.Generator().manual_seed(42))
    assert torch.equal(x + 1, y)
    assert x.shape == (5, 4)
    with pytest.raises(ValueError):
        sample_batch(tokens[:4], 4, 1, torch.Generator())


def test_attention_numerical_and_mask():
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    k = q.clone()
    v = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    out, weights = scaled_attention(q, k, v)
    assert out.shape == (3, 2) and weights.shape == (3, 3)
    assert torch.equal(weights.triu(1), torch.zeros(3, 3))
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    assert torch.allclose(out[0], v[0])
    p = 1 / (1 + math.exp(1 / math.sqrt(2)))
    assert torch.allclose(out[1], p * v[0] + (1 - p) * v[1])
    _, bidirectional = scaled_attention(q, k, v, causal=False)
    assert torch.all(bidirectional > 0)


def test_head_split_merge():
    x = torch.arange(2 * 5 * 24).reshape(2, 5, 24)
    assert split_heads(x, 3).shape == (2, 3, 5, 8)
    assert torch.equal(merge_heads(split_heads(x, 3)), x)


def test_position_formulas():
    pe = sinusoidal(4, 5)
    assert torch.equal(pe[0], torch.tensor([0., 1., 0., 1., 0.]))
    assert pe[1, 0].item() == pytest.approx(math.sin(1))
    assert pe[1, 2].item() == pytest.approx(math.sin(10000 ** (-2 / 5)))
    x = torch.ones(2, 3, 4, 8)
    rotated = apply_rope(x)
    assert rotated.shape == x.shape
    assert torch.equal(rotated[:, :, 0], x[:, :, 0])
    assert torch.allclose(rotated.square().sum(-1), x.square().sum(-1), atol=1e-5)
    assert rotated[0, 0, 1, 0].item() == pytest.approx(math.cos(1) - math.sin(1))


@pytest.mark.parametrize("position", ["sinusoidal", "learned", "rope"])
def test_model_causal_loss_and_gradients(position):
    torch.manual_seed(42)
    config = tiny(position=position)
    model = GPT(config).eval()
    x = torch.randint(9, (2, 8))
    changed = x.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 9
    a, loss, attentions = model(x, x, return_attention=True)
    b, _, _ = model(changed)
    assert a.shape == (2, 8, 9)
    assert attentions[0].shape == (2, 3, 8, 8)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert torch.allclose(loss, F.cross_entropy(a.reshape(-1, 9), x.reshape(-1)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    block = TransformerBlock(config)
    assert block(torch.randn(2, 8, 24))[0].shape == (2, 8, 24)


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        tiny(n_heads=5)
    with pytest.raises(ValueError):
        tiny(d_model=21, position="rope")


def test_locked_registry():
    root = Path(__file__).resolve().parents[1]
    baseline, training, _ = load_experiment(root, "E01", 65)
    assert (baseline.d_model, baseline.n_heads, baseline.n_layers, baseline.context_length) == (192, 6, 4, 128)
    expected = {"E02": {}, "E03": {"position": "rope"}, "E04": {"position": "learned"},
                "E05": {"activation": "relu"}, "E06": {"n_heads": 4}, "E07": {"n_heads": 8},
                "E08": {"n_layers": 6}, "E09": {"n_layers": 8}, "E10": {"mlp_expansion": 2},
                "E11": {"context_length": 64}, "E12": {"context_length": 256},
                "E13": {"d_model": 144}, "E14": {"d_model": 240}}
    assert len(list((root / "experiments/configs").glob("*.json"))) == 14
    for eid, overrides in expected.items():
        model, settings, variant = load_experiment(root, eid, 65)
        delta = {k: v for k, v in model.to_dict().items() if baseline.to_dict()[k] != v}
        assert delta == overrides
        assert settings == training
        assert variant["tokenizer"] == ("bpe" if eid == "E02" else "character")
        assert training["target_tokens_per_update"] % model.context_length == 0


def test_tiny_overfit():
    torch.manual_seed(42)
    model = GPT(tiny())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]])
    y = torch.tensor([[1, 2, 0, 1, 2, 0, 1, 2]])
    initial = model(x, y)[1].item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.15


def test_generation_deterministic():
    model = GPT(tiny())
    ids = torch.tensor([[0, 1]])
    assert torch.equal(generate(model, ids, 10, seed=123), generate(model, ids, 10, seed=123))
    assert generate(model, ids, 10, greedy=True).shape == (1, 12)
    assert model.training
