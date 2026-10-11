"""Storage-only fixtures: no synthetic research metrics or model qualification."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
from nano_lm.artifacts import file_hash,digest

spec=importlib.util.spec_from_file_location('pilot_rehydrate',Path(__file__).parents[1]/'scripts/pilot/rehydrate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root):
    checkpoint=root/'runs/fixture/checkpoints/step_000001.pt';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'storage fixture bytes, not a model')
    snapshot=root/'snapshots/fixture_step1';snapshot.mkdir(parents=True)
    manifest={'schema_version':1,'run_id':'fixture','step':1,'config_hash':'fixture','files':{'checkpoints/step_000001.pt':file_hash(checkpoint)}}
    manifest['hash']=digest(manifest);(snapshot/'export_manifest.json').write_text(json.dumps(manifest))
    return checkpoint,snapshot


def test_expanded_input_rehydrated_only_in_writable_copy(tmp_path):
    source=tmp_path/'mounted';_,snapshot=fixture(source)
    target=tmp_path/'writable';shutil.copytree(source,target)
    records=module.complete_snapshots(target)
    assert len(records)==1
    assert not (snapshot/'checkpoints/step_000001.pt').exists()
    assert (target/'snapshots/fixture_step1/checkpoints/step_000001.pt').read_bytes()==b'storage fixture bytes, not a model'


def test_corrupt_source_checkpoint_rejected(tmp_path):
    checkpoint,_=fixture(tmp_path);checkpoint.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum mismatch'):module.complete_snapshots(tmp_path)
