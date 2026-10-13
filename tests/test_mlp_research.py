from pathlib import Path
from scripts.pilot.mlp_audit import audit


def test_E10_MLP_numerical_reference_and_gradients():
    result=audit(Path(__file__).resolve().parents[1])
    assert result['parameters']==1213440 and result['parameter_reduction']==591360
    assert result['MLP_parameters_per_block']==[148032]*4
    assert len(result['checks'])==14 and len(result['projection_weight_gradients'])==4
