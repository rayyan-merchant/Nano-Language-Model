from pathlib import Path
from scripts.pilot.learned_audit import audit


def test_canonical_learned_positions_numerical_audit():
    result=audit(Path(__file__).resolve().parents[1])
    assert result['status']=='passed' and len(result['checks'])==13
    assert result['parameters']==1829376 and result['additional_parameters']==24576
