"""Serving contracts use explicitly injected fixtures, not research-weight evidence."""
import json
import time
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from api.catalog import catalog
from api.main import create_app
from api.registry import Registry, ModelUnavailable
from api.schemas import GenerateRequest
from tools.package_models import export


class FixtureRegistry:
    """Small deterministic fixture reachable only by explicit test injection."""
    def __init__(self, delay=0):
        torch.manual_seed(123)
        self.tokenizer = Registry(Path('missing')).tokenizer
        self.entries = catalog()
        self.device = 'cpu'
        self.delay = delay
        self.models = {key: GPT(ModelConfig(1024, d_model=12, n_heads=3, n_layers=1)).eval() for key in self.entries}
    def load(self, key):
        time.sleep(self.delay)
        return self.models[key]
    def describe(self):
        return [{**value, 'availability':'ready'} for value in self.entries.values()]


def test_selected_identity_and_parameter_contracts():
    entries = catalog()
    assert [(e['checkpoint_step'], e['parameters']) for e in entries.values()] == [(5000,2173056),(3000,3062784),(1750,6046560)]
    for entry in entries.values():
        assert GPT(ModelConfig(**entry['architecture'])).parameter_count() == entry['parameters']


def test_wrong_hash_rejected_before_pickle_deserialization(tmp_path):
    file = tmp_path/'untrusted.pt';file.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        export('b01',file,tmp_path/'export')


def test_health_missing_models_and_sanitized_failure(tmp_path):
    with TestClient(create_app(Registry(tmp_path))) as client:
        assert client.get('/health').status_code == 200
        assert all(m['availability']=='unavailable' for m in client.get('/v1/models').json()['models'])
        assert client.get('/ready').status_code == 503
        response=client.post('/v1/generate',json={'prompt':'ROMEO:'})
        assert response.status_code==503 and str(tmp_path) not in response.text
        result=client.post('/v1/compare',json={'prompt':'ROMEO:'}).json()
        assert len(result['results'])==3 and all(r['status']=='unavailable' for r in result['results'])


@pytest.mark.parametrize('change',[{'model_id':'e01'},{'max_new_tokens':257},{'max_new_tokens':True},
    {'temperature':0},{'temperature':3},{'seed':-1},{'prompt':''},{'prompt':' '},{'prompt':'x'*4001},
    {'checkpoint_path':'private'},{'prompt':'\ud800'}])
def test_strict_request_validation(change):
    with TestClient(create_app(FixtureRegistry())) as client:
        assert client.post('/v1/generate',content=json.dumps({'prompt':'ROMEO:',**change}),
                           headers={'Content-Type':'application/json'}).status_code==422


def test_seeded_greedy_switching_context_and_compare():
    with TestClient(create_app(FixtureRegistry())) as client:
        for greedy in [False, True]:
            body={'model_id':'e02','prompt':'ROMEO:\n','max_new_tokens':12,'seed':42,'greedy':greedy}
            a=client.post('/v1/generate',json=body).json();b=client.post('/v1/generate',json=body).json()
            assert a['full_output']==b['full_output'] and a['generated_tokens']==12
            assert a['full_output'].startswith(body['prompt'])
            assert a['replacement_characters']==('\ufffd' in a['continuation'])
        result=client.post('/v1/generate',json={'prompt':'ROMEO: '*150,'max_new_tokens':1}).json()
        assert result['context_cropped'] and result['effective_context']==128
        comparison=client.post('/v1/compare',json={'prompt':'é','max_new_tokens':3}).json()
        assert [r['model_id'] for r in comparison['results']]==['e02','b01','f01']
        assert all(r['status']=='completed' and r['result']['seed']==42 for r in comparison['results'])


def test_size_content_type_rate_and_timeout():
    with TestClient(create_app(FixtureRegistry(),rate_limit=1)) as client:
        response=client.post('/v1/generate',content='x'*32769,headers={'Content-Type':'application/json','Origin':'http://localhost:3000'})
        assert response.status_code==413
        assert response.headers['access-control-allow-origin']=='http://localhost:3000'
        assert client.post('/v1/generate',json={'prompt':'x'}).status_code==429
    with TestClient(create_app(FixtureRegistry())) as client:
        assert client.post('/v1/generate',content='x').status_code==415
    with TestClient(create_app(FixtureRegistry(delay=.2),timeout=.01)) as client:
        assert client.post('/v1/generate',json={'prompt':'x','max_new_tokens':1}).status_code==504
        assert client.post('/v1/generate',json={'prompt':'x','max_new_tokens':1}).status_code==429


def test_corrupt_serving_identity_never_loads(tmp_path):
    directory=tmp_path/'b01';directory.mkdir()
    (directory/'metadata.json').write_text(json.dumps({**catalog()['b01'],'checkpoint_step':4500}))
    registry=Registry(tmp_path)
    with pytest.raises(ModelUnavailable):registry.load('b01')
    assert registry.describe()[1]['availability']=='unavailable'


def test_frontend_result_contract_matches_server_fields():
    import re
    from api.schemas import GenerationResult, CompareResult, Settings
    text=Path('frontend/lib/api.ts').read_text(encoding='utf-8')
    for name, schema in [('Generation',GenerationResult),('Settings',Settings)]:
        declaration=re.search(r'export type '+name+r' = \{(.*?)\n\};',text,re.S).group(1)
        fields=set(re.findall(r'^\s*(\w+)\??:',declaration,re.M))
        assert fields == set(schema.model_fields)


def test_serving_size_mismatch_rejected_before_hash_or_tensor_loading(tmp_path,monkeypatch):
    directory=tmp_path/'b01';directory.mkdir()
    (directory/'model.safetensors').write_bytes(b'x')
    metadata={**catalog()['b01'],'format_version':1,'precision':'float32',
              'artifact_bytes':999999999,'artifact_sha256':'0'*64}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr('api.registry.file_hash',lambda _:pytest.fail('Size must be checked before reading weights'))
    with pytest.raises(ModelUnavailable):Registry(tmp_path).load('b01')
