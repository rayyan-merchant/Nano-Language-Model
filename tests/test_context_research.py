from pathlib import Path
import pytest
from scripts.pilot.context_audit import audit, token_budget
from nano_lm.config import load_experiment
from nano_lm.train import TrainSettings

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('experiment,context,accumulation,windows',[('E01',128,8,32),('E11',64,16,64)])
def test_resolved_context_token_budget(experiment,context,accumulation,windows):
    model, defaults, _ = load_experiment(ROOT,experiment,65)
    settings=TrainSettings(eval_every=250,checkpoint_every=500)
    settings.validate(model.context_length)
    grouping=token_budget(model,settings)
    assert grouping=={'context_length':context,'micro_batch_size':4,'accumulation':accumulation,
                      'sampled_windows_per_update':windows,'predicted_positions_per_update':4096}
    assert defaults == load_experiment(ROOT,'E01',65)[1]

def test_context64_numerical_and_evaluator_reference():
    result=audit(ROOT)
    assert result['parameters']==1804800 and len(result['checks'])==14
    assert result['seed42_initial_trainable_weights_equal_E01']
