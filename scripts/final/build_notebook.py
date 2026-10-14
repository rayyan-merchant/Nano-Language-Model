"""F01-only notebooks reuse the proven isolated provisioning bootstrap."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import uuid
import subprocess
import hashlib
from scripts.pilot.build_notebook import build as provision_template


def build(output, commit, mode, source=None, diagnostic_commit=None):
    if mode not in {'development','development-recovery','research','research-recovery'}:
        raise ValueError('Unknown F01 mode')
    if mode != 'development' and (not source or not source.startswith('rayyannmerchant/')):
        raise ValueError('Published same-account prerequisite is mandatory')
    if mode == 'development' and source:
        raise ValueError('Fresh development must not mount another run')
    if diagnostic_commit is not None:
        if mode!='research-recovery' or len(diagnostic_commit)!=40 or any(c not in '0123456789abcdef' for c in diagnostic_commit):
            raise ValueError('Additional pure-inference diagnostics require a pinned helper commit and research-recovery mode')
    output=Path(output)
    # Only reuse package provisioning; replace the entire execution command.
    # No E14 experiment is executed or enabled by this builder.
    with contextlib.redirect_stdout(io.StringIO()):
        provision_template(output,commit,experiment='E14',development_only=True)
    notebook=json.loads((output/'pilot.ipynb').read_text(encoding='utf-8'))
    metadata=json.loads((output/'kernel-metadata.json').read_text(encoding='utf-8'))
    previous=metadata['id'];kernel=f'rayyannmerchant/nano-f01-{mode}-20261014-{uuid.uuid4().hex[:8]}'
    code=''.join(notebook['cells'][1]['source'])
    before, _, _ = code.partition("print('START PILOT ORCHESTRATION',flush=True)")
    if not before or 'scripts/pilot/run.py' in before:
        raise ValueError('Provisioning template execution boundary changed')
    # Kaggle documents 12h CPU/GPU sessions. Use a conservative 10h pipeline
    # envelope, reserving 2h for provisioning/publication, not an API guarantee.
    code=before.replace(previous,kernel).replace("'NANO_SESSION_LIMIT_SECONDS':'7200'", "'NANO_SESSION_LIMIT_SECONDS':'36000'")+f"""print('F01 {mode}: exact source '+COMMIT,flush=True)
result=subprocess.run([str(PY),'-m','scripts.final.workflow','--root',str(ROOT),'--output',str(OUT/'final'),'--mode',{mode!r}],cwd=ROOT,env=env)
if result.returncode:raise RuntimeError('F01 workflow failed; preserve genuine partial evidence')
"""
    notebook['metadata']['nano_lm_provenance']={'training_source_commit':commit,
        'launcher_builder_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        'mode':mode,'scope':'Launcher uses unchanged qualified provisioning, then invokes canonical package wrapper with visible logs'}
    if diagnostic_commit:
        helper_bytes=subprocess.check_output(['git','show',f'{diagnostic_commit}:scripts/final/reproduce_generation.py'],cwd=Path(__file__).resolve().parents[2])
        helper_hash=hashlib.sha256(helper_bytes).hexdigest()
        notebook['metadata']['nano_lm_provenance'].update(diagnostic_helper_commit=diagnostic_commit,diagnostic_helper_sha256=helper_hash)
        code+=f"""import urllib.request, hashlib
helper=Path('/tmp/f01-generation-reproduction.py')
raw=urllib.request.urlopen('https://raw.githubusercontent.com/rayyan-merchant/Nano-Language-Model/{diagnostic_commit}/scripts/final/reproduce_generation.py').read()
assert hashlib.sha256(raw).hexdigest()=={helper_hash!r}
helper.write_bytes(raw)
result=subprocess.run([str(PY),str(helper),'--root',str(ROOT),'--output',str(OUT/'generation_reproduction'),'--helper-commit',{diagnostic_commit!r}],cwd=ROOT,env={{**env,'PYTHONPATH':str(ROOT)}})
if result.returncode:raise RuntimeError('Fixed generation reproduction failed; preserve original research artifacts')
"""
    notebook['cells'][0]['source']=[f'# F01 seed42 {mode}\nOne authorized fresh research trajectory, target 10000 updates and 40 full validations. Development evidence is separate. Recovery adds no research updates. Source: {commit}.']
    notebook['cells'][1]['source']=code.splitlines(True)
    metadata.update(id=kernel,title=kernel.split('/')[1],kernel_sources=[source] if source else [])
    (output/'pilot.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return {'kernel':kernel,'source_commit':commit,'mode':mode,'folder':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--mode',choices=['development','development-recovery','research','research-recovery'],required=True);p.add_argument('--source')
    p.add_argument('--diagnostic-commit')
    a=p.parse_args();print(json.dumps(build(a.output,a.commit,a.mode,a.source,a.diagnostic_commit)))
