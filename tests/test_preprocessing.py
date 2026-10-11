import math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from nano_lm.bpe import BPETokenizer, encode_with_bytes, from_artifact
from nano_lm.tokenizer import CharacterTokenizer
from nano_lm.prepare import prepare, validate
from nano_lm.evaluate import evaluate, evaluation_tokens, metrics


@pytest.mark.parametrize("kind", ["character", "bpe"])
def test_lossless_utf8_and_serialization(kind):
    text = "hello café 🙂\n" * 30
    t = CharacterTokenizer.fit(text) if kind == "character" else BPETokenizer.fit(text, 300)
    t2 = from_artifact(t.artifact())
    ids, lengths = encode_with_bytes(t2, text)
    assert t.decode(ids) == text
    assert t.checksum == t2.checksum
    assert sum(lengths) == len(text.encode())
    if kind == "bpe":
        # Byte alphabet supports unseen UTF-8 bytes without fitting held-out text.
        assert t.decode(t.encode("雪")) == "雪"


def test_manifest_and_train_only_fit(tmp_path):
    text = "abc café\n" * 100
    directory = tmp_path / "prepared"
    manifest = prepare(directory, text.encode(), bpe_vocab_size=300)
    assert manifest == validate(directory)
    assert sum(p["bytes"] for p in manifest["partitions"].values()) == len(text.encode())
    assert all(t["fit_partition"] == "train" for t in manifest["tokenizers"].values())
    train = (directory / "train.txt").read_bytes().decode()
    fitted = CharacterTokenizer.fit(train)
    assert fitted.checksum == manifest["tokenizers"]["character"]["hash"]
    (directory / "validation.txt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Partition"):
        validate(directory)


class Uniform(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.))
        self.config = SimpleNamespace(context_length=8)
        self.vocab = vocab

    def forward(self, x):
        return torch.zeros(*x.shape, self.vocab, device=x.device) + self.p, None, []


@pytest.mark.parametrize("kind", ["character", "bpe"])
@pytest.mark.parametrize("mode", ["sliding", "blocks"])
def test_bpb_independent_uniform_likelihood(kind, mode):
    text = "abc café 🙂\n" * 20
    t = CharacterTokenizer.fit(text) if kind == "character" else BPETokenizer.fit(text, 300)
    vocab = len(t.characters) if kind == "character" else t.vocab_size
    model = Uniform(vocab).train()
    ids, prefix, lengths, prefix_bytes, total = evaluation_tokens(t, text, 16)
    result = evaluate(model, t, text, 16, batch_size=3, mode=mode)
    count = len(ids) - prefix
    assert result["nll_sum"] == pytest.approx(count * math.log(vocab), rel=1e-6)
    assert result["bpb"] == pytest.approx(count * math.log2(vocab) / sum(lengths), rel=1e-6)
    assert result["scored_bytes"] == len(text.encode()) - prefix_bytes
    assert result["scored_tokens"] == count
    assert result["perplexity"] == pytest.approx(vocab, rel=1e-6)
    assert model.training


def test_bpb_rejects_empty_coverage():
    with pytest.raises(ValueError):
        metrics(3., 0, 4)
