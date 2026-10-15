"""Retrieval safety fixtures are not trained-model execution evidence."""
import io
import json
import re
from pathlib import Path
import urllib.error

import pytest
from tools import submission_models as retrieval


def test_selected_manifest_matches_accepted_packages():
    release = retrieval.manifest()
    assert release['revision'] == '719a7f3a4955225a9818e872b08c1503de601a81'
    for mid, spec in release['models'].items():
        reference = json.loads((retrieval.ROOT / f'docs/deployment/evidence/{mid}-package.json').read_text(encoding='utf-8'))
        for key in ['parameters', 'checkpoint_step', 'architecture', 'original_checkpoint_sha256',
                    'training_commit', 'artifact_sha256', 'artifact_bytes', 'tokenizer_hash']:
            assert spec[key] == reference[key]


def test_existing_destination_is_never_requested_or_overwritten(tmp_path):
    output = tmp_path / 'models'
    output.mkdir()
    evidence = output / 'evidence'
    evidence.write_text('preserve')
    with pytest.raises(FileExistsError):
        retrieval.download(output, opener=lambda *_a, **_k: pytest.fail('must not download'))
    assert evidence.read_text() == 'preserve'


@pytest.mark.parametrize('kind', ['partial', 'corrupt', 'oversized', 'http'])
def test_failed_download_never_publishes_release(tmp_path, kind):
    spec = retrieval.manifest()['models']['e02']
    def open_response(request, timeout):
        assert request.get_header('Authorization') is None
        assert retrieval.manifest()['revision'] in request.full_url
        if kind == 'http':
            raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, None)
        if kind == 'corrupt':
            data = b'x' * spec['metadata_bytes']
        elif kind == 'oversized':
            data = b'x' * (spec['metadata_bytes'] + 1)
        else:
            data = b'x'
        return io.BytesIO(data)
    output = tmp_path / 'models'
    with pytest.raises((ValueError, RuntimeError)):
        retrieval.download(output, opener=open_response)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_missing_and_symlink_artifacts_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing or unsafe'):
        retrieval.checksums(tmp_path)


def test_demo_uses_canonical_modules_and_all_fixed_prompts():
    from tools import submission_demo
    from nano_lm.generate import generate
    assert submission_demo.generate is generate
    assert len(submission_demo.PROMPTS) == 5
    assert submission_demo.PROMPTS[:3] == ['To be, or not to be', 'ROMEO:\n', 'JULIET:\n']


def test_installed_modules_may_live_outside_checkout(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools import submission_demo
    outside = tmp_path / 'external-environment/site-packages/nano_lm/model.py'
    monkeypatch.setattr(submission_demo.importlib, 'import_module', lambda _name: SimpleNamespace(__file__=str(outside)))
    monkeypatch.setattr(submission_demo.subprocess, 'check_output', lambda *_a, **_k: '')
    receipt = submission_demo.environment()
    assert all(Path(p).is_absolute() for p in receipt['modules'].values())
    assert str(outside.resolve()) in receipt['modules'].values()


def test_submission_notebook_is_clean_pinned_and_compiles():
    notebook = json.loads((retrieval.ROOT / 'notebooks/NanoLM_Submission_Demo.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(code) == 13
    for cell in code:
        assert cell['execution_count'] is None and cell['outputs'] == []
        compile(''.join(cell['source']), 'submission-cell', 'exec')
    source = '\n'.join(''.join(c['source']) for c in code)
    assert re.search(r'SOURCE = "[0-9a-f]{40}"', source)
    assert 'nano_lm.train' not in source and 'auth login' not in source
    assert 'tools.submission_models' in source and 'pip", "check' in source
