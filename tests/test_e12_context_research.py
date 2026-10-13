from pathlib import Path
from scripts.pilot.e12_context_audit import audit, token_budget
from nano_lm.config import load_experiment
from nano_lm.train import TrainSettings

ROOT=Path(__file__).resolve().parents[1]

def test_e12_context256_numerical_reference():
    result=audit(ROOT)
    assert result['parameters']==1804800 and len(result['checks'])==14
    assert result['model']['context_length']==256
    assert result['evaluation_reference']['scored_bytes']==44

def test_e12_accumulation4_and_only_context_override():
    model,training,variant=load_experiment(ROOT,'E12',65)
    assert variant['model_overrides']=={'context_length':256}
    assert training==load_experiment(ROOT,'E01',65)[1]
    assert token_budget(model,TrainSettings())=={'context_length':256,'micro_batch_size':4,'accumulation':4,'sampled_windows_per_update':16,'predicted_positions_per_update':4096}


def test_e12_notebook_scope_and_persistent_gate(tmp_path):
    import pytest
    from scripts.pilot.build_e12_notebook import build
    source='a'*40
    for phase in ['preflight','research','recover-final']:
        notebook=build(source,phase,tmp_path/(phase+'.ipynb'))
        code=''.join(notebook['cells'][1]['source'])
        compile(code,'notebook','exec')
        assert 'e12_source' in code and 'NANO_SOURCE_COMMIT' in code
        if phase=='research':
            assert code.index("'recover-smoke'")<code.index("'research'")<code.index("'finish'")
            assert 'E12_smoke_bundle.zip' in code
        else:
            assert "'research','--root'" not in code
    with pytest.raises(ValueError):build(source,'E11',tmp_path/'bad.ipynb')
    with pytest.raises(ValueError):build('main','research',tmp_path/'bad.ipynb')


def test_e12_resume_only_accepts_incomplete_single_lineage(monkeypatch,tmp_path):
    import importlib.util,sys,pytest
    import scripts.pilot.e12_context_audit as context
    sys.modules['e12_context_audit']=context
    spec=importlib.util.spec_from_file_location('e12_workflow',ROOT/'scripts/pilot/e12_workflow.py')
    workflow=importlib.util.module_from_spec(spec);sys.modules['e12_workflow']=workflow;spec.loader.exec_module(workflow)
    spec=importlib.util.spec_from_file_location('e12_kaggle',ROOT/'scripts/pilot/e12_kaggle.py')
    kaggle=importlib.util.module_from_spec(spec);sys.modules['e12_kaggle']=kaggle;spec.loader.exec_module(kaggle)
    spec=importlib.util.spec_from_file_location('e12_resume',ROOT/'scripts/pilot/e12_resume.py')
    resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)
    monkeypatch.setattr(resume,'verify_snapshot',lambda p:{'run_id':'E12_run','step':int(p.name),'files':{}})
    assert resume.select_snapshot([tmp_path/'250',tmp_path/'500'])[2]['step']==500
    for paths in [[],[tmp_path/'0'],[tmp_path/'5000']]:
        with pytest.raises(ValueError):resume.select_snapshot(paths)
    monkeypatch.setattr(resume,'verify_snapshot',lambda p:{'run_id':p.name,'step':250,'files':{}})
    with pytest.raises(ValueError):resume.select_snapshot([tmp_path/'first',tmp_path/'second'])
