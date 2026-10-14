"""B01-only launcher reusing qualified provisioning, never F01 execution."""
import argparse
import json
from pathlib import Path
from scripts.final.build_notebook import build as provision
from scripts.balanced.protocol import STOP_POLICY


def build(output,commit,mode,source=None):
    result=provision(output,commit,mode,source)
    output=Path(output);nb=json.loads((output/'pilot.ipynb').read_text(encoding='utf-8'))
    meta=json.loads((output/'kernel-metadata.json').read_text(encoding='utf-8'))
    old=result['kernel'];new=old.replace('nano-f01-','nano-b01-')
    code=''.join(nb['cells'][1]['source']).replace(old,new).replace('scripts.final.workflow','scripts.balanced.workflow')
    code=code.replace("OUT/'final'","OUT/'balanced'").replace('F01 workflow','B01 workflow').replace('F01 '+mode,'B01 '+mode)
    assert 'scripts.final.workflow' not in code and 'scripts/pilot/run.py' not in code
    nb['cells'][1]['source']=code.splitlines(True)
    nb['cells'][0]['source']=[f'# B01 {mode}\nSeed42; at most5000updates; frozen validation early stopping; no test likelihood. Exact source:{commit}.']
    nb['metadata']['nano_lm_provenance'].update(experiment='B01',maximum_updates=5000,early_stopping=STOP_POLICY)
    meta.update(id=new,title=new.split('/')[1])
    (output/'pilot.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return dict(kernel=new,source_commit=commit,mode=mode,folder=str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--mode',choices=['development','development-recovery','research','research-recovery'],required=True)
    p.add_argument('--source');a=p.parse_args();print(json.dumps(build(a.output,a.commit,a.mode,a.source)))
