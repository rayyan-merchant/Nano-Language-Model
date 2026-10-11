import json
from pathlib import Path
import pytest
from nano_lm.artifacts import digest
from nano_lm.bpe import from_artifact, vocabulary_size
from nano_lm.config import ModelConfig
from nano_lm.prepare import prepare
from nano_lm.train import TrainSettings, train
from nano_lm.storage import export_run, verify_snapshot, restore_snapshot
from nano_lm.results import collect, validate_run
from nano_lm.experiments import audit

ROOT = Path(__file__).resolve().parents[1]


def make_run(tmp_path):
    data = tmp_path / "data"
    prepare(data, ("abc def\n" * 100).encode(), bpe_vocab_size=280)
    t = from_artifact(json.loads((data / "character.json").read_text(encoding="utf-8")))
    cfg = ModelConfig(vocabulary_size(t), d_model=24, n_heads=3, n_layers=1, context_length=8, dropout=0.1)
    settings = TrainSettings(run_kind="smoke", tokens_per_update=16, micro_batch_size=1,
                             eval_every=1, checkpoint_every=1, conditioning_bytes=8, eval_max_bytes=32)
    run, ckpt, summary = train(ROOT, data, "E01", settings, 2, tmp_path / "runs", model_override=cfg,
                              export_root=tmp_path / "exports")
    return data, cfg, settings, run, ckpt


def test_export_restore_resume_and_corruption(tmp_path):
    data, cfg, settings, run, ckpt = make_run(tmp_path)
    snapshot = export_run(run, tmp_path / "exports")
    verified = verify_snapshot(snapshot)
    assert verified["step"] == 2
    restored = restore_snapshot(snapshot, tmp_path / "restored")
    restored_checkpoint = restored / "checkpoints" / ckpt.name
    _, final, summary = train(ROOT, data, "E01", settings, 3, tmp_path / "restored",
                             resume=restored_checkpoint, model_override=cfg)
    assert final.exists() and summary["step"] == 3
    with pytest.raises(FileExistsError):
        restore_snapshot(snapshot, tmp_path / "restored")
    (snapshot / "tokenizer.json").write_text("corrupted")
    with pytest.raises(ValueError, match="Corrupt"):
        verify_snapshot(snapshot)


def test_aggregation_rejects_incomplete_smoke_and_metadata_tampering(tmp_path):
    _, _, _, run, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match="research"):
        validate_run(run, ROOT, 2)
    rows, aggregate = collect([run, run], ROOT, 2, allow_checks=True, require_complete=False)
    assert len(rows) == 1 and aggregate[0]["n_seeds"] == 1
    assert aggregate[0]["validation_bpb_sd"] is None
    with pytest.raises(ValueError, match="Complete study"):
        collect([run], ROOT, 2, allow_checks=True)
    meta = json.loads((run / "metadata.json").read_text())
    meta["resolved"]["training"]["learning_rate"] = 0.05
    (run / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run(run, ROOT, 2, allow_checks=True)


def test_all_fourteen_instantiation_and_shapes(tmp_path):
    data = tmp_path / "data"
    prepare(data, ("abc def\n" * 100).encode(), bpe_vocab_size=280)
    rows = audit(ROOT, data)
    assert len(rows) == 14
    assert all(row["logits"][0] == 1 and row["logits"][2] == row["vocab_size"] for row in rows)
    assert rows[10]["logits"][1] == 64 and rows[11]["logits"][1] == 256
