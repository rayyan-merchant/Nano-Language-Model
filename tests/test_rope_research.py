from pathlib import Path
from scripts.pilot.rope_audit import audit


def test_locked_rope_research_path():
    result=audit(Path(__file__).resolve().parents[1])
    assert result['status']=='passed' and len(result['checks'])==13
