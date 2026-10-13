"""Durable resume selection rejects corruption, mixed lineage and completed runs."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/pilot'))
from e07_resume import select_snapshot
from nano_lm.artifacts import digest, file_hash


def snapshot(root,step,run='E07_seed42_example'):
    p=root/f'{run}_{step}';p.mkdir()
    state=p/'state';state.write_bytes(b'complete-state')
    m={'schema_version':1,'run_id':run,'step':step,'config_hash':'test','files':{'state':file_hash(state)}}
    m['hash']=digest(m);(p/'export_manifest.json').write_text(json.dumps(m))
    return p


def test_latest_snapshot_requires_single_incomplete_verified_lineage(tmp_path):
    early=snapshot(tmp_path,250);latest=snapshot(tmp_path,2750)
    _,selected,m=select_snapshot([early,latest])
    assert selected==latest and m['step']==2750
    other=snapshot(tmp_path,3000,'different-run')
    for paths in ([early,other],[snapshot(tmp_path,5000)],[]):
        try: select_snapshot(paths)
        except ValueError: pass
        else: raise AssertionError('Invalid resume lineage accepted')


def test_corrupt_snapshot_cannot_be_selected_for_resume(tmp_path):
    p=snapshot(tmp_path,2750);(p/'state').write_bytes(b'corrupt')
    try:select_snapshot([p])
    except ValueError:pass
    else:raise AssertionError('Corrupt durable checkpoint accepted')
