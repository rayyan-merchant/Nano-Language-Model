"""Build a unique private E01 pilot notebook from a clean committed checkout."""
import argparse
import json
from pathlib import Path
import uuid


def build(output, commit, source_kernel=None, source_dataset=None, analysis_dataset=None, experiment='E01', restore_only=False):
    if experiment not in {'E01','E03','E04','E06'}:raise ValueError('Unsupported experiment')
    if restore_only and not source_kernel:raise ValueError('Restore-only requires published notebook source')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An exact source SHA is required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if source_kernel and source_dataset:raise ValueError('Choose one recovery source')
    phase = 'recovery' if source_kernel or source_dataset else 'pilot'
    if analysis_dataset and (not source_dataset or not analysis_dataset.startswith('rayyannmerchant/')):
        raise ValueError('Analysis reuse requires authorized private dataset recovery')
    kernel = f'rayyannmerchant/nano-{experiment.lower()}-{phase}-20261012-' + uuid.uuid4().hex[:8]
    bootstrap = f'''from pathlib import Path
import os, sys, subprocess, json
ROOT=Path('/tmp/nano-source'); OUT=Path('/kaggle/working')
COMMIT={commit!r}
env={{**os.environ,'UV_PYTHON_INSTALL_DIR':'/tmp/pilot-python','UV_CACHE_DIR':'/tmp/pilot-cache','UV_NATIVE_TLS':'true','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/pilot-mpl','MPLBACKEND':'Agg','NANO_KERNEL_ID':{kernel!r},'NANO_SESSION_LIMIT_SECONDS':'7200'}}
def command(args,cwd=None):
    print('EXEC',args,flush=True)
    with (OUT/'bootstrap.log').open('a') as log:
        result=subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Bootstrap failed: '+str(args))
command(['git','clone','https://github.com/rayyan-merchant/Nano-Language-Model.git',str(ROOT)])
command(['git','checkout','--detach',COMMIT],ROOT)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
command([sys.executable,'-m','pip','install','--target','/tmp/pilot-tools','uv==0.12.7'])
uv=next(p for p in [Path('/tmp/pilot-tools/bin/uv'),Path('/tmp/pilot-tools/uv/uv')] if p.is_file())
command([str(uv),'python','install','3.11.14'])
command([str(uv),'venv','--seed','--python','3.11.14','/tmp/pilot-venv'])
PY=Path('/tmp/pilot-venv/bin/python')
command([str(uv),'pip','install','--python',str(PY),'torch==2.5.1+cu124','--index-url','https://download.pytorch.org/whl/cu124'])
freeze=(ROOT/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines()
lock=OUT/'qualified-lock.txt'
lock.write_text('\\n'.join(x for x in freeze if not x.startswith('-e '))+'\\n')
command([str(uv),'pip','install','--python',str(PY),'-r',str(lock)])
command([str(uv),'pip','install','--python',str(PY),'-e',str(ROOT),'--no-deps'])
command([str(PY),'-m','pip','check'])
print('START PILOT ORCHESTRATION',flush=True)
result=subprocess.run([str(PY),str(ROOT/'scripts/pilot/run.py'),'--root',str(ROOT),'--output',str(OUT/'pilot'),'--experiment',{experiment!r}],cwd=ROOT,env=env)
if result.returncode: raise RuntimeError('Pilot failed; preserve outputs and inspect failure evidence')
'''
    if source_kernel or source_dataset:
        if not (source_kernel or source_dataset).startswith('rayyannmerchant/'):
            raise ValueError('Recovery must use the authorized account')
        bootstrap = bootstrap.replace("scripts/pilot/run.py", "scripts/pilot/postprocess_recovery.py").replace("OUT/'pilot'", "OUT/'pilot_recovered'")
    if restore_only:
        bootstrap=bootstrap.replace('scripts/pilot/postprocess_recovery.py','scripts/pilot/recover.py').replace("OUT/'pilot_recovered'","OUT/'recovery_audit'").replace(",'--experiment',"+repr(experiment), '')
    elif source_kernel or source_dataset:
        bootstrap=bootstrap.replace(",'--experiment',"+repr(experiment), '')
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
        {'id':'pilot-scope','cell_type':'markdown','metadata':{},'source':[f'# Authorized {experiment} seed42 {phase}\nPilot: exactly 5000 updates. Recovery: zero additional updates. Source {commit}. No other experiments or extension.']},
        {'id':'pilot-execution','cell_type':'code','metadata':{},'source':bootstrap.splitlines(True),'execution_count':None,'outputs':[]}]}
    (output/'pilot.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    metadata={'id':kernel,'title':kernel.split('/')[1],'code_file':'pilot.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':([source_dataset]+([analysis_dataset] if analysis_dataset else [])) if source_dataset else [],'competition_sources':[],'kernel_sources':[source_kernel] if source_kernel else []}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'kernel':kernel,'source_commit':commit,'folder':str(output)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--commit',required=True);p.add_argument('--source-kernel');p.add_argument('--source-dataset');p.add_argument('--analysis-dataset');p.add_argument('--experiment',choices=['E01','E03','E04','E06'],default='E01');p.add_argument('--restore-only',action='store_true')
    args=p.parse_args();build(args.output,args.commit,args.source_kernel,args.source_dataset,args.analysis_dataset,args.experiment,args.restore_only)
