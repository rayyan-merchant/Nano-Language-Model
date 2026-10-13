from pathlib import Path
from scripts.pilot.heads_audit import audit


def test_canonical_four_head_attention_numerical_audit():
    result = audit(Path(__file__).resolve().parents[1])
    assert result['status'] == 'passed' and len(result['checks']) == 13
    assert result['parameters'] == 1804800
    assert result['all_initial_parameter_values_match_E01']
