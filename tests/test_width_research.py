from pathlib import Path
from scripts.pilot.width_audit import audit
from scripts.pilot.context_audit import token_budget
from nano_lm.config import load_experiment
from nano_lm.train import TrainSettings

ROOT=Path(__file__).resolve().parents[1]

def test_width144_independent_numerical_references():
    result=audit(ROOT)
    assert result['parameters']==1021824 and result['parameter_reduction']==782976
    assert result['MLP_parameters_per_block']==[166608]*4 and len(result['checks'])==20

def test_E13_and_baseline_grouping_and_sole_width_override():
    base,defaults,_=load_experiment(ROOT,'E01',65)
    candidate,training,_=load_experiment(ROOT,'E13',65)
    assert training==defaults
    assert {k for k in base.to_dict() if base.to_dict()[k]!=candidate.to_dict()[k]}=={'d_model'}
    settings=TrainSettings(eval_every=250,checkpoint_every=500)
    for model in [base,candidate]:
        grouping=token_budget(model,settings)
        assert grouping['context_length']==128 and grouping['accumulation']==8
        assert grouping['sampled_windows_per_update']==32 and grouping['predicted_positions_per_update']==4096
