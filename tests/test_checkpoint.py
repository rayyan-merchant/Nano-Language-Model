import json
import os
import subprocess
import sys
import pytest
import torch
from nano_lm.checkpoint import load_for_inference, read_checkpoint


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_resume_matches_uninterrupted_and_cli(tmp_path):
    full, partial = tmp_path / "full", tmp_path / "partial"
    run("-m", "nano_lm.smoke", "--run-dir", full, "--steps", 6)
    run("-m", "nano_lm.smoke", "--run-dir", partial, "--steps", 3)
    run("-m", "nano_lm.smoke", "--run-dir", partial, "--steps", 6,
        "--resume", partial / "step_000003.pt")
    a, _ = read_checkpoint(full / "step_000006.pt")
    b, _ = read_checkpoint(partial / "step_000006.pt")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["scheduler"] == b["scheduler"]
    assert torch.equal(a["rng"]["sampling"], b["rng"]["sampling"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    ha = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
    hb = [json.loads(line) for line in (partial / "metrics.jsonl").read_text().splitlines()]
    assert [r["training_loss"] for r in ha] == [r["training_loss"] for r in hb]
    model, tokenizer = load_for_inference(partial / "step_000006.pt")
    assert model.config.vocab_size == len(tokenizer.characters)
    output = run("-m", "nano_lm.generate", "--checkpoint", partial / "step_000006.pt",
                 "--prompt", "To be", "--max-new-tokens", 3, "--greedy")
    assert output.startswith("To be") and "parameters=" in output
    failed = subprocess.run([sys.executable, "-m", "nano_lm.smoke", "--run-dir", str(full)],
                            capture_output=True)
    assert failed.returncode != 0


def test_checkpoint_rejects_bad_tokenizer(tmp_path):
    directory = tmp_path / "run"
    run("-m", "nano_lm.smoke", "--run-dir", directory, "--steps", 1)
    path = directory / "step_000001.pt"
    payload, _ = read_checkpoint(path)
    payload["tokenizer_hash"] = "wrong"
    bad = directory / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(bad)


def test_unicode_cli_under_legacy_console_encoding(tmp_path):
    from nano_lm.checkpoint import save_checkpoint
    from nano_lm.config import ModelConfig
    from nano_lm.model import GPT
    from nano_lm.tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer.fit("aλ")
    model = GPT(ModelConfig(vocab_size=2, d_model=8, n_heads=2, n_layers=1,
                            context_length=4))
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    path = tmp_path / "unicode.pt"
    save_checkpoint(path, model, tokenizer, optimizer, scheduler,
                    torch.Generator().manual_seed(42), 0, {})
    result = subprocess.run([sys.executable, "-m", "nano_lm.generate",
                             "--checkpoint", str(path), "--prompt", "λ",
                             "--max-new-tokens", "0"], capture_output=True,
                            env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode("utf-8").startswith("λ")
