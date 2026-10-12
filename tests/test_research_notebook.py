import json
import pytest
from scripts.pilot.build_notebook import build


def code(folder):
    return ''.join(json.loads((folder/'pilot.ipynb').read_text())['cells'][1]['source'])


@pytest.mark.parametrize('experiment',['E03','E04','E06','E08','E09'])
def test_positional_notebook_and_recovery_cannot_train_other_variants(tmp_path,experiment):
    sha='a'*40
    build(tmp_path/'research',sha,experiment=experiment)
    source=code(tmp_path/'research');compile(source,'research','exec')
    assert f"'--experiment','{experiment}'" in source and "'MPLBACKEND':'Agg'" in source
    m=json.loads((tmp_path/'research/kernel-metadata.json').read_text())
    assert m['id'].startswith(f'rayyannmerchant/nano-{experiment.lower()}-') and m['is_private']
    build(tmp_path/'recovery',sha,source_kernel=m['id'],experiment=experiment,restore_only=True)
    source=code(tmp_path/'recovery');compile(source,'recovery','exec')
    assert 'scripts/pilot/recover.py' in source and 'scripts/pilot/run.py' not in source
    assert "'--experiment'" not in source
