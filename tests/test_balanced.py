import math
from pathlib import Path
import pytest
from nano_lm.model import GPT
from scripts.balanced.protocol import configuration, parameter_breakdown, provenance, PARAMETERS
from scripts.balanced.stopping import StoppingRule
from scripts.balanced.state import scientific_state

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_configuration_and_historical_parity():
    model = GPT(configuration(ROOT))
    assert model.parameter_count() == parameter_breakdown()['total'] == PARAMETERS
    assert len(model.blocks) == 6 and model.config.d_model == 192
    assert provenance(ROOT)['unchanged_F01_and_pilot_evidence']


def test_patience_accumulates_before_eligibility():
    rule = StoppingRule()
    for step in range(250, 2000, 250):
        assert rule.observe(step, 2.)['stopping_reason'] is None
    assert rule.counter == 6
    assert rule.observe(2000, 2.)['stopping_reason'] == 'early_stop'
    with pytest.raises(ValueError): rule.observe(2250, 1.)


def test_strict_threshold_and_absolute_minimum_are_separate():
    rule = StoppingRule(); rule.observe(250, 2.)
    boundary = 2. - .001
    row = rule.observe(500, boundary)
    assert not row['meaningful_improvement'] and rule.best['step'] == 500
    assert rule.reference == 2. and rule.counter == 1
    row = rule.observe(750, math.nextafter(boundary, -math.inf))
    assert row['meaningful_improvement'] and rule.counter == 0


def test_cumulative_small_improvements_and_budget():
    rule = StoppingRule(); rule.observe(250, 2.)
    rule.observe(500, 1.9996); rule.observe(750, 1.9992)
    assert rule.counter == 2 and rule.best['step'] == 750
    assert rule.observe(1000, 1.9988)['meaningful_improvement']
    for step in range(1250, 5001, 250): rule.observe(step, 2.-step/10000)
    assert rule.reason == 'budget_exhausted' and rule.best['step'] == 5000


@pytest.mark.parametrize('step,bpb', [(500,2.),(250,float('nan')),(250,float('inf')),(250,-1.)])
def test_invalid_measurements_fail(step,bpb):
    with pytest.raises(ValueError): StoppingRule().observe(step,bpb)


def test_continuation_projection_preserves_every_scientific_history_value():
    original={'step':2,'training_bytes':99,'train_seconds':1.,'history':[
        {'training_loss':2.,'gradient_norm':.7,'train_seconds':1.,'validation':{'bpb':2.1,'scored_bytes':55514}}]}
    projected=scientific_state(original)
    assert projected=={'step':2,'training_bytes':99,'history':[
        {'training_loss':2.,'gradient_norm':.7,'validation':{'bpb':2.1,'scored_bytes':55514}}]}


def test_canonical_committed_tokenizer_is_not_refitted():
    import json
    from nano_lm.bpe import from_artifact
    from scripts.balanced.protocol import BPE
    tok=from_artifact(json.loads((ROOT/'assets/tokenizers/bpe_canonical.json').read_text(encoding='utf-8')))
    assert tok.checksum==BPE and tok.vocab_size==1024
    assert tok.decode(tok.encode('ROMEO:\né €'))=='ROMEO:\né €'


def test_b01_notebook_scope_and_source_gate(tmp_path):
    import json
    from scripts.balanced.build_notebook import build
    result=build(tmp_path/'dev','a'*40,'development')
    nb=json.loads((tmp_path/'dev/pilot.ipynb').read_text())
    code=''.join(nb['cells'][1]['source'])
    assert 'scripts.balanced.workflow' in code and 'scripts.final.workflow' not in code
    assert result['kernel'].startswith('rayyannmerchant/nano-b01-development-')
    assert nb['metadata']['nano_lm_provenance']['maximum_updates']==5000
    with pytest.raises(ValueError):build(tmp_path/'blocked','a'*40,'research')
