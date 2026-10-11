from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from nano_lm.prepare import prepare
from nano_lm.bpe import from_artifact, vocabulary_size
from nano_lm.config import ModelConfig
from nano_lm.checkpoint import read_checkpoint, load_for_inference
from nano_lm.train import TrainSettings, train
from nano_lm.generate import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data"
    prepare(path, ("To be, or not to be.\n" * 100).encode(), bpe_vocab_size=280)
    return path


def config(dataset, kind="character", dropout=0.1):
    tokenizer = from_artifact(json.loads((dataset / f"{kind}.json").read_text(encoding="utf-8")))
    return ModelConfig(vocabulary_size(tokenizer), d_model=24, n_heads=3,
                       n_layers=1, context_length=8, dropout=dropout)


def settings(**kwargs):
    return replace(TrainSettings(run_kind="smoke", tokens_per_update=32, micro_batch_size=2,
                                eval_every=2, checkpoint_every=1, conditioning_bytes=16,
                                eval_max_bytes=64), **kwargs)


def test_real_engine_resume_and_interrupted_journal(dataset, tmp_path):
    cfg = config(dataset)
    full, a, _ = train(ROOT, dataset, "E01", settings(), 4, tmp_path / "full", model_override=cfg)
    partial, mid, _ = train(ROOT, dataset, "E01", settings(), 2, tmp_path / "partial", model_override=cfg)
    # A failed/unsaved attempt is retained as an event, never treated as committed metrics.
    with (partial / "events.jsonl").open("a") as f:
        f.write('{"event":"uncommitted_test_attempt","step":3}\n')
    _, b, summary = train(ROOT, dataset, "E01", settings(), 4, tmp_path / "partial",
                          resume=mid, model_override=cfg)
    pa, _ = read_checkpoint(a)
    pb, _ = read_checkpoint(b)
    assert all(torch.equal(pa["model"][k], pb["model"][k]) for k in pa["model"])
    assert pa["scheduler"] == pb["scheduler"]
    assert torch.equal(pa["rng"]["sampling"], pb["rng"]["sampling"])
    assert torch.equal(pa["rng"]["torch"], pb["rng"]["torch"])
    for key in ("training_tokens", "training_bytes", "best_loss"):
        assert pa["training_state"][key] == pb["training_state"][key]
    assert pa["training_state"]["training_tokens"] == 128
    assert pa["training_state"]["training_bytes"] == 128  # ASCII character tokens.
    assert "uncommitted_test_attempt" in (partial / "events.jsonl").read_text()
    for sa, sb in zip(pa["training_state"]["history"], pb["training_state"]["history"]):
        assert sa["training_loss"] == sb["training_loss"]
        assert sa.get("validation") == sb.get("validation")
    model, tokenizer = load_for_inference(b)
    output = generate(model, torch.tensor([tokenizer.encode("To be")]), 3, seed=42)
    assert tokenizer.decode(output[0].tolist()).startswith("To be")
    with pytest.raises(ValueError, match="mismatch"):
        train(ROOT, dataset, "E01", settings(learning_rate=0.01), 5, tmp_path,
              resume=b, model_override=cfg)


def test_gradient_accumulation_equivalence(dataset, tmp_path):
    cfg = config(dataset, dropout=0)
    _, a, _ = train(ROOT, dataset, "E01", settings(micro_batch_size=1), 1, tmp_path / "a", model_override=cfg)
    _, b, _ = train(ROOT, dataset, "E01", settings(micro_batch_size=4), 1, tmp_path / "b", model_override=cfg)
    pa, _ = read_checkpoint(a)
    pb, _ = read_checkpoint(b)
    assert torch.equal(pa["rng"]["sampling"], pb["rng"]["sampling"])
    # Compare optimizer first moments: these are the scaled accumulated gradients.
    # Adam can magnify near-zero reduction noise (e.g. the softmax-invariant key bias),
    # so bitwise parameter equality is not a valid accumulation criterion.
    for index in pa["optimizer"]["state"]:
        ga = pa["optimizer"]["state"][index]["exp_avg"]
        gb = pb["optimizer"]["state"][index]["exp_avg"]
        assert torch.allclose(ga, gb, atol=1e-8, rtol=1e-5), index
    ma, _ = load_for_inference(a)
    mb, _ = load_for_inference(b)
    x = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]])
    assert torch.allclose(ma(x)[0], mb(x)[0], atol=1e-5)


def test_bpe_training_checkpoint_and_bytes(dataset, tmp_path):
    cfg = config(dataset, "bpe")
    _, path, _ = train(ROOT, dataset, "E02", settings(), 2, tmp_path, model_override=cfg)
    payload, tokenizer = read_checkpoint(path)
    assert payload["training_state"]["training_bytes"] > payload["training_state"]["training_tokens"]
    assert tokenizer.artifact()["kind"] == "bpe"
    assert payload["training_state"]["history"][-1]["validation"]["scored_bytes"] == 48


def test_clear_invalid_device_and_budget():
    with pytest.raises(ValueError):
        settings(tokens_per_update=31).validate(8)
    with pytest.raises(ValueError):
        settings(precision="float16").validate(8)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            settings(device="cuda").validate(8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")
def test_cuda_amp_resume(dataset, tmp_path):
    cfg = config(dataset)
    gpu = settings(device="cuda", precision="float16")
    _, a, _ = train(ROOT, dataset, "E01", gpu, 4, tmp_path / "a", model_override=cfg)
    _, mid, _ = train(ROOT, dataset, "E01", gpu, 2, tmp_path / "b", model_override=cfg)
    _, b, _ = train(ROOT, dataset, "E01", gpu, 4, tmp_path / "b", resume=mid, model_override=cfg)
    pa, _ = read_checkpoint(a)
    pb, _ = read_checkpoint(b)
    assert pa["scaler"] == pb["scaler"]
    assert all(torch.equal(pa["model"][k], pb["model"][k]) for k in pa["model"])
