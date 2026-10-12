"""Numerical E07 control audit against the canonical shared package."""
from pathlib import Path
from scripts.pilot.heads_audit import audit


def test_e07_eight_head_controls():
    result=audit(Path(__file__).resolve().parents[1])
    assert result['status']=='passed'
    assert len(result['numerical_attention_probes'])==4
