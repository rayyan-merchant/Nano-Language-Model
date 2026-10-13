"""Independent CPU saved-output archive audit, comparison and small-file export.

No trainer or optimizer is called. Large source archives are verified first;
immutable bounded parts preserve their exact bytes for unreliable local links.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from inspect_archive import inspect
from compare_heads import compare
from compare_generation_grid import describe
from diagnostics import diagnostics
from nano_lm.prepare import prepare
from nano_lm.train import environment


def run(root,out):
    out.mkdir(parents=True,exist_ok=False);retrieved=out/'retrieved';retrieved.mkdir()
    env=environment(root)
    assert not env['git_dirty'] and env['code_hash']=='cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'
    assert env['python']=='3.11.14' and env['torch']=='2.5.1+cu124'
    folders={}
    for eid in ['E01','E10']:
        sources=list(Path('/kaggle/input').rglob(eid+'_seed42_pilot.zip'));assert len(sources)==1
        archive=retrieved/(eid+'_seed42_pilot.zip');manifest=retrieved/(eid+'_manifest.json')
        shutil.copyfile(sources[0],archive);shutil.copyfile(sources[0].parent/'artifact_manifest.json',manifest)
        folder=out/(eid+'_verified');inspect(archive,manifest,folder);folders[eid]=folder
    data=Path('/tmp/audit-data');prepare(data)
    diagnostics(folders['E10'],data/'train.txt',out/'checkpoint_diagnostics.json')
    compare(folders['E01'],folders['E10'],root,out/'comparison','E10')
    describe(root,folders['E01']/'analysis/generation.json',folders['E10']/'analysis/generation.json',out/'generation_quality.json','E10')
    source=retrieved/'E10_seed42_pilot.zip';parts=out/'archive_parts';parts.mkdir();rows=[]
    with source.open('rb') as f:
        index=0
        while data:=f.read(4*2**20):
            name=f'part_{index:03d}.bin';(parts/name).write_bytes(data)
            rows.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
    manifest=json.loads((retrieved/'E10_manifest.json').read_text())
    (parts/'parts_manifest.json').write_text(json.dumps({'archive_bytes':manifest['archive_bytes'],'archive_sha256':manifest['archive_sha256'],'parts':rows,'policy':'Ordered immutable4MiB parts of independently SHA/CRC/all-file verified original archive; no training/model change'},indent=2)+'\n')
    # Export only small genuine report material and verification receipts.
    bundle=out/'analysis_bundle.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in folders['E10'].rglob('*'):
            if path.is_file() and (path.parent==folders['E10'] or 'analysis' in path.relative_to(folders['E10']).parts) and path.suffix!='.pt':z.write(path,'verified/'+path.relative_to(folders['E10']).as_posix())
        for name in ['comparison','checkpoint_diagnostics.json','generation_quality.json']:
            path=out/name
            for file in path.rglob('*') if path.is_dir() else [path]:
                if file.is_file():z.write(file,file.relative_to(out).as_posix())
    receipt={'status':'independent_saved_output_CPU_archive_audit_and_comparison_passed','environment':env,'additional_optimizer_updates':0,'E10_archive_sha256':manifest['archive_sha256'],'E10_archive_bytes':manifest['archive_bytes'],'part_count':len(rows),'analysis_bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'analysis_bundle_bytes':bundle.stat().st_size,'scope':'Fresh CPU-only Kaggle runtime retrieved full E01/E10 published archives; SHA256, CRC, all-file checks and best/final inference; no research training'}
    (out/'remote_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CPU ARCHIVE AUDIT COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
