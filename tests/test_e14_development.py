"""Synthetic control fixtures only; these do not qualify real cloud storage."""
import json
from pathlib import Path
import pytest
from scripts.pilot import development


def test_research_gate_requires_unique_exact_committed_recovery(tmp_path,monkeypatch):
    receipt_path=tmp_path/'development_recovery.json'
    monkeypatch.setattr(Path,'rglob',lambda self,pattern:[receipt_path] if receipt_path.exists() else [])
    monkeypatch.setattr(development,'environment',lambda root:{'git_commit':'a'*40})
    with pytest.raises(RuntimeError):development.require_recovery_receipt(tmp_path,tmp_path)
    receipt={'status':'published_E14_development_fresh_CUDA_full_state_passed','source_commit':'a'*40,
             'parameters':2808960,'step_before':1,'step_after':1,'additional_optimizer_updates':0,
             'exact_state':{k:True for k in ['model','optimizer','scheduler','scaler','rng','training_state']}}
    receipt_path.write_text(json.dumps(receipt))
    development.require_recovery_receipt(tmp_path,tmp_path)
    assert json.loads((tmp_path/'independent_development_recovery_gate.json').read_text())==receipt
    receipt['source_commit']='b'*40;receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):development.require_recovery_receipt(tmp_path,tmp_path)
    receipt['source_commit']='a'*40;receipt['exact_state']['rng']=False;receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):development.require_recovery_receipt(tmp_path,tmp_path)
    receipt['exact_state']['rng']=True;receipt['additional_optimizer_updates']=1;receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):development.require_recovery_receipt(tmp_path,tmp_path)
