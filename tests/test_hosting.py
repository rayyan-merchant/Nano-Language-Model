"""Hosting transport boundaries; bounded fixtures are not research-model evidence."""
import hashlib
import json
from pathlib import Path
import pytest
from api import hosting


@pytest.fixture
def fixture_manifest(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.bin').write_bytes(b'explicit-transport-test-fixture')
    data = (source / 'model.bin').read_bytes()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'e02/model.safetensors': {
        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}))
    monkeypatch.setattr(hosting, 'MANIFEST', manifest)
    return source / 'model.bin'


def test_private_pinned_download_and_cached_restart(tmp_path, fixture_manifest):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return fixture_manifest
    destination = hosting.prepare('owner/private-models', 'a'*40, tmp_path/'cache', 'test-secret', download)
    assert (destination/'e02/model.safetensors').read_bytes() == fixture_manifest.read_bytes()
    assert calls[0]['revision'] == 'a'*40 and calls[0]['token'] == 'test-secret'
    assert hosting.prepare('owner/private-models', 'a'*40, tmp_path/'cache', 'test-secret', download) == destination
    assert len(calls) == 1
    (destination/'e02/model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        hosting.prepare('owner/private-models', 'a'*40, tmp_path/'cache', 'test-secret', download)


@pytest.mark.parametrize('repo,revision,token', [
    ('https://external/path', 'a'*40, 'test'), ('owner/repo', 'main', 'test'),
    ('owner/repo', 'a'*40, ''), ('owner/repo/extra', 'a'*40, 'test')])
def test_rejects_mutable_or_invalid_source(tmp_path, repo, revision, token):
    with pytest.raises(ValueError):
        hosting.prepare(repo, revision, tmp_path, token)


def test_bad_download_does_not_publish_partial_model(tmp_path, fixture_manifest):
    fixture_manifest.write_bytes(b'bad payload')
    with pytest.raises(ValueError):
        hosting.prepare('owner/repo', 'b'*40, tmp_path/'cache', 'test', lambda **_: fixture_manifest)
    assert list((tmp_path/'cache').iterdir()) == []


def test_manifest_contains_only_six_exact_selected_inference_files():
    manifest = json.loads(hosting.MANIFEST.read_text())
    assert set(manifest['files']) == {f'{m}/{f}' for m in ['e02','b01','f01']
                                      for f in ['metadata.json','model.safetensors']}
    for m in ['e02','b01','f01']:
        receipt = json.loads(Path(f'docs/deployment/evidence/{m}-package.json').read_text(encoding='utf-8'))
        assert manifest['files'][f'{m}/model.safetensors']['sha256'] == receipt['artifact_sha256']


@pytest.mark.parametrize('identity,private', [('other-person', True), ('owner', False)])
def test_publisher_rejects_wrong_account_or_public_repository(tmp_path, monkeypatch, identity, private):
    from tools import publish_hf_models as publisher
    class FakeAPI:
        def whoami(self): return {'name': identity}
        def create_repo(self, *args, **kwargs): assert kwargs['private'] is True
        def repo_info(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(private=private)
        def create_commit(self, *args, **kwargs):
            pytest.fail('No file upload is allowed on wrong identity or public repository')
    monkeypatch.setattr(publisher, 'HfApi', FakeAPI)
    monkeypatch.setattr(publisher, 'verify', lambda *args: None)
    with pytest.raises(ValueError):
        publisher.publish('owner', 'owner/models', tmp_path, tmp_path/'receipt.json')
    assert not (tmp_path/'receipt.json').exists()


def test_vercel_bundle_is_verified_once_and_uses_canonical_registry(tmp_path, monkeypatch):
    from api import bundled_runtime
    calls = []
    monkeypatch.setattr(bundled_runtime, 'verify', lambda *args: calls.append(args))
    monkeypatch.setattr(bundled_runtime.Registry, 'load', lambda self, key: key)
    registry = bundled_runtime.BundledRegistry(tmp_path)
    assert registry.load('e02') == 'e02'
    assert registry.load('f01') == 'f01'
    assert len(calls) == 1 and registry.verified


def test_vercel_invalid_private_bundle_is_sanitized(tmp_path, monkeypatch):
    from api import bundled_runtime
    from api.registry import ModelUnavailable
    def fail(*args): raise RuntimeError('private filesystem details fixture')
    monkeypatch.setattr(bundled_runtime, 'verify', fail)
    registry = bundled_runtime.BundledRegistry(tmp_path)
    with pytest.raises(ModelUnavailable) as error:
        registry.load('b01')
    assert 'filesystem' not in str(error.value)
    assert not registry.verified


def test_vercel_builder_emits_supported_glob_and_cpu_only_profile(tmp_path, monkeypatch):
    from tools import prepare_vercel_backend as builder
    source = tmp_path/'weights'
    (source/'e02').mkdir(parents=True)
    (source/'e02/model.safetensors').write_bytes(b'explicit-private-fixture')
    data = (source/'e02/model.safetensors').read_bytes()
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'files': {'e02/model.safetensors': {
        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}))
    monkeypatch.setattr(builder, 'MANIFEST', manifest)
    def fixture_build(destination):
        destination.mkdir()
        (destination/'requirements-studio.txt').write_text('fastapi==0.143.1\n')
        return {'source_commit': 'a'*40}
    monkeypatch.setattr(builder, 'build', fixture_build)
    destination = tmp_path/'private-function'
    receipt = builder.prepare(destination, source)
    config = json.loads((destination/'vercel.json').read_text())
    assert isinstance(config['functions']['app.py']['includeFiles'], str)
    assert 'private_models' in config['functions']['app.py']['includeFiles']
    assert (destination/'private_models/e02/model.safetensors').read_bytes() == data
    assert not (destination/'public').exists()
    assert 'torch==2.5.1+cpu' in (destination/'requirements.txt').read_text()
    assert 'matplotlib' not in (destination/'requirements.txt').read_text()
    assert not receipt['public_static_model_files']
