"""Measure E12 development snapshot/export overhead; no optimizer updates."""
import argparse,json,time,zipfile
from pathlib import Path
from nano_lm.storage import export_run,verify_snapshot
from nano_lm.artifacts import file_hash
from nano_lm.checkpoint import read_checkpoint


def measure(out):
    run=next((out/'preflight/smoke_runs').iterdir())
    checkpoint=max((run/'checkpoints').glob('step_*.pt'))
    payload,_=read_checkpoint(checkpoint)
    assert payload['config']['context_length']==256 and payload['metadata']['resolved']['experiment_id']=='E12'
    assert payload['metadata']['resolved']['training']['run_kind']=='smoke'
    began=time.perf_counter();snapshot=export_run(run,out/'export_benchmark_snapshots');elapsed=time.perf_counter()-began
    manifest=verify_snapshot(snapshot);size=sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file())
    archive=out/'export_benchmark.zip';began=time.perf_counter()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in snapshot.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(snapshot).as_posix())
    archive_seconds=time.perf_counter()-began
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    record={'scope':'Real E12 development smoke immutable export and archive; no research updates','export_seconds':elapsed,'snapshot_bytes':size,'export_bytes_per_second':size/elapsed,'archive_seconds':archive_seconds,'archive_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),'snapshot_manifest_hash':manifest['hash']}
    (out/'export_throughput.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);measure(p.parse_args().output)
