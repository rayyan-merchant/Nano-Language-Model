"""Snapshot archive references must recover exact bytes and reject corrupt weights."""
import json
from pathlib import Path
from nano_lm.artifacts import digest, file_hash
from scripts.pilot.verify_e07_archive import reconstruct_snapshots


def fixture(root):
    cp=root/'runs/run/checkpoints/step_000250.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'checkpoint-state-bytes')
    snapshot=root/'snapshots/snapshot250';snapshot.mkdir(parents=True)
    journal=snapshot/'events.jsonl';journal.write_bytes(b'{"event":"step","step":250}\n')
    m={'schema_version':1,'run_id':'run','step':250,'config_hash':'example','files':{'events.jsonl':file_hash(journal),'checkpoints/step_000250.pt':file_hash(cp)}}
    m['hash']=digest(m);(snapshot/'export_manifest.json').write_text(json.dumps(m))
    return cp,m


def test_snapshot_metadata_and_shared_weights_recover_exact_bytes(tmp_path):
    archive=tmp_path/'archive';cp,m=fixture(archive)
    manifests=reconstruct_snapshots(archive,'runs/run/',tmp_path/'restored')
    assert manifests==[m]
    assert (tmp_path/'restored/snapshot250/checkpoints/step_000250.pt').read_bytes()==cp.read_bytes()
    assert (tmp_path/'restored/snapshot250/events.jsonl').read_bytes()==(archive/'snapshots/snapshot250/events.jsonl').read_bytes()


def test_corrupt_shared_weight_fails_before_preservation_claim(tmp_path):
    archive=tmp_path/'archive';cp,_=fixture(archive);cp.write_bytes(b'corrupt')
    try:
        reconstruct_snapshots(archive,'runs/run/',tmp_path/'restored')
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt shared checkpoint was accepted')
