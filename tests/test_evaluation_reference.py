import pytest
import torch
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.tokenizer import CharacterTokenizer
from nano_lm.evaluate import evaluate


def test_validation_matches_one_full_causal_forward():
    text = "abcdabdcabcd"
    t = CharacterTokenizer.fit(text)
    model = GPT(ModelConfig(4, d_model=24, n_heads=3, n_layers=1, context_length=32, dropout=0.3)).eval()
    ids = torch.tensor([t.encode(text)])
    with torch.no_grad():
        logits = model(ids[:, :-1])[0]
        # Prefix is the first two bytes; reference uses one full forward, not
        # the evaluator's per-target sliding-window implementation.
        nll = torch.nn.functional.cross_entropy(logits[:, 1:].reshape(-1, 4),
                                                ids[:, 2:].reshape(-1), reduction="sum").item()
    a = evaluate(model, t, text, conditioning_bytes=2, batch_size=1)
    b = evaluate(model, t, text, conditioning_bytes=2, batch_size=8)
    assert a["nll_sum"] == pytest.approx(nll, rel=1e-6)
    assert b["nll_sum"] == pytest.approx(nll, rel=1e-6)
    assert a["scored_tokens"] == a["scored_bytes"] == 10


def test_dropout_train_and_eval_behavior():
    model = GPT(ModelConfig(4, d_model=24, n_heads=3, n_layers=1, context_length=8, dropout=0.5))
    x = torch.zeros((1, 8), dtype=torch.long)
    model.train()
    assert not torch.equal(model(x)[0], model(x)[0])
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0])
