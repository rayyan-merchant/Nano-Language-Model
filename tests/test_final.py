"""F01 authorization, historical integrity, numerical and launcher regression."""
import json
from pathlib import Path
import pytest
import torch
from nano_lm.config import load_experiment
from nano_lm.model import GPT
from nano_lm.train import TrainSettings
from scripts.final.protocol import configuration, PARAMETERS, parameter_breakdown, provenance, AUDITED
from scripts.final.build_notebook import build
from scripts.final.audit import reference_rotation
from nano_lm.positions import apply_rope

ROOT=Path(__file__).resolve().parents[1]


def test_f01_configuration_and_baseline_regression():
    config=configuration(ROOT)
    assert GPT(config).parameter_count()==PARAMETERS==parameter_breakdown()['total']
    base,defaults,variant=load_experiment(ROOT,'E01',65)
    assert base.d_model==192 and base.n_layers==4 and base.position=='sinusoidal'
    assert defaults['optimizer_steps']==5000 and defaults['seeds']==[42,123,456]
    settings=TrainSettings();assert settings.tokens_per_update//(4*config.context_length)==8
    assert 10000*settings.tokens_per_update==40960000


def test_frozen_scientific_files_and_new_aggregate():
    result=provenance(ROOT)
    assert result['audited_pre_F01_aggregate_hash']==AUDITED
    assert result['unchanged_historical_scientific_files']


def test_independent_rope_and_wrong_reference():
    torch.manual_seed(42);x=torch.randn(1,6,128,40)
    torch.testing.assert_close(apply_rope(x),reference_rotation(x),rtol=2e-5,atol=3e-5)
    assert not torch.allclose(apply_rope(x),reference_rotation(x,True))


def test_f01_launcher_scope(tmp_path):
    commit='a'*40
    with pytest.raises(ValueError):build(tmp_path/'blocked',commit,'research')
    result=build(tmp_path/'dev',commit,'development')
    nb=json.loads((tmp_path/'dev/pilot.ipynb').read_text())
    source=''.join(nb['cells'][1]['source'])
    assert 'scripts.final.workflow' in source and 'scripts/pilot/run.py' not in source
    assert result['kernel'].startswith('rayyannmerchant/nano-f01-development-')
    assert "'NANO_SESSION_LIMIT_SECONDS':'36000'" in source
    assert 'torch==2.5.1+cu124' in source
    with pytest.raises(FileExistsError):build(tmp_path/'dev',commit,'development')
    with pytest.raises(ValueError):build(tmp_path/'wrong',commit,'research','another-account/notebook')
    with pytest.raises(ValueError):build(tmp_path/'wrong-diag',commit,'research','rayyannmerchant/notebook',diagnostic_commit='a'*40)
