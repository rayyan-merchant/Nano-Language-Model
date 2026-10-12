from pathlib import Path
from scripts.pilot.relu_audit import audit


def test_canonical_relu_numerical_and_initialization_audit():
    result = audit(Path(__file__).resolve().parents[1])
    assert result['status'] == 'passed'
    assert result['parameters'] == 1804800
    assert result['all_initial_E01_parameter_tensors_equal']
    assert len(result['numerical_activation_probes']) == 4
    assert all(r['nonzero_negative_input_gradients'] == 0 for r in result['actual_MLP_gradient_flow'])
