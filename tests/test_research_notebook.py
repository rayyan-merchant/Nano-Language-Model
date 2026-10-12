import json
from scripts.pilot.build_notebook import build


def code(folder):
    return ''.join(json.loads((folder/'pilot.ipynb').read_text())['cells'][1]['source'])


def test_e03_notebook_and_recovery_cannot_train_other_variants(tmp_path):
    sha='a'*40
    build(tmp_path/'research',sha,experiment='E03')
    source=code(tmp_path/'research');compile(source,'research','exec')
    assert "'--experiment','E03'" in source and "'MPLBACKEND':'Agg'" in source
    m=json.loads((tmp_path/'research/kernel-metadata.json').read_text())
    assert m['id'].startswith('rayyannmerchant/nano-e03-') and m['is_private']
    build(tmp_path/'recovery',sha,source_kernel=m['id'],experiment='E03',restore_only=True)
    source=code(tmp_path/'recovery');compile(source,'recovery','exec')
    assert 'scripts/pilot/recover.py' in source and 'scripts/pilot/run.py' not in source
    assert "'--experiment'" not in source
