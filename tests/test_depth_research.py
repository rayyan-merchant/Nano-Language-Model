from pathlib import Path
from scripts.pilot.depth_audit import audit


def test_six_block_depth_numerical_contract():
    result = audit(Path(__file__).resolve().parents[1])
    assert len(result['checks']) == 13
    assert result['additional_parameters'] == 889728
    assert all(x['gradient_norm'] > 0 for x in result['block_gradients'])


def test_eight_block_depth_numerical_contract():
    result = audit(Path(__file__).resolve().parents[1], experiment='E09')
    assert len(result['checks']) == 15
    assert result['block_count'] == 8 and result['parameters'] == 3584256
    assert result['additional_parameters'] == 1779456
    assert all(x['gradient_norm'] > 0 for x in result['block_gradients'])
