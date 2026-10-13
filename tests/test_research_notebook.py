import json
import pytest
from scripts.pilot.build_notebook import build


def code(folder):
    return ''.join(json.loads((folder/'pilot.ipynb').read_text())['cells'][1]['source'])


@pytest.mark.parametrize('experiment',['E03','E04','E06','E08','E09','E10','E11','E13','E14'])
def test_positional_notebook_and_recovery_cannot_train_other_variants(tmp_path,experiment):
    sha='a'*40
    build(tmp_path/'research',sha,experiment=experiment,qualification_kernel='rayyannmerchant/e14-gate/1' if experiment=='E14' else None)
    source=code(tmp_path/'research');compile(source,'research','exec')
    assert f"'--experiment','{experiment}'" in source and "'MPLBACKEND':'Agg'" in source
    m=json.loads((tmp_path/'research/kernel-metadata.json').read_text())
    assert m['id'].startswith(f'rayyannmerchant/nano-{experiment.lower()}-') and m['is_private']
    build(tmp_path/'recovery',sha,source_kernel=m['id'],experiment=experiment,restore_only=True)
    source=code(tmp_path/'recovery');compile(source,'recovery','exec')
    assert 'scripts/pilot/recover.py' in source and 'scripts/pilot/run.py' not in source
    assert "'--experiment'" not in source


def test_E14_development_and_recovery_are_closed_before_research(tmp_path):
    sha='b'*40
    with pytest.raises(ValueError):build(tmp_path/'blocked',sha,experiment='E14')
    build(tmp_path/'development',sha,experiment='E14',development_only=True)
    source=code(tmp_path/'development');compile(source,'development','exec')
    assert "'--development-only'" in source
    build(tmp_path/'dev-recovery',sha,source_kernel='rayyannmerchant/e14-development/1',experiment='E14',development_recovery=True)
    source=code(tmp_path/'dev-recovery');compile(source,'dev-recovery','exec')
    assert 'scripts/pilot/development.py' in source and 'scripts/pilot/run.py' not in source
    assert "'--experiment'" not in source
