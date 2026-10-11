"""Create private qualification jobs; core source is always a pinned main checkout."""
import argparse
import json
import uuid
from pathlib import Path

COMMIT = "b2ef1ca842ff6b0794b802f1baae58226732674f"
BOOTSTRAP = '''from pathlib import Path
import os, sys, subprocess, json, time
OUT=Path('/kaggle/working')
ROOT=Path('/tmp/nano-source')
COMMIT='COMMIT_VALUE'
env={**os.environ,'UV_PYTHON_INSTALL_DIR':'/tmp/qual-python','UV_CACHE_DIR':'/tmp/qual-cache','UV_NATIVE_TLS':'true','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/qual-mpl'}
log=OUT/'bootstrap.log'
def command(args,cwd=None):
    print('BOOTSTRAP',args,flush=True)
    with log.open('a') as f:
        result=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Bootstrap failed {args}; see bootstrap.log')
command(['git','clone','https://github.com/rayyan-merchant/Nano-Language-Model.git',str(ROOT)])
command(['git','checkout','--detach',COMMIT],ROOT)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
tools=Path('/tmp/qual-tools')
command([sys.executable,'-m','pip','install','--target',str(tools),'uv==0.12.7'])
uv=next(p for p in [tools/'bin/uv',tools/'uv/uv'] if p.is_file())
command([str(uv),'python','install','3.11.14'])
command([str(uv),'venv','--seed','--python','3.11.14','/tmp/qual-venv'])
PY=Path('/tmp/qual-venv/bin/python')
command([str(uv),'pip','install','--python',str(PY),'torch==2.5.1+cu124','--index-url','https://download.pytorch.org/whl/cu124'])
command([str(uv),'pip','install','--python',str(PY),'-r',str(ROOT/'requirements.txt')])
command([str(uv),'pip','install','--python',str(PY),'-e',str(ROOT),'--no-deps'])
command([str(PY),'-m','pip','check'])
command([str(PY),'-c',"import sys,torch,numpy; assert sys.version_info[:2]==(3,11); assert str(torch.__version__)=='2.5.1+cu124'; assert torch.version.cuda=='12.4' and torch.cuda.is_available(); assert numpy.__version__=='1.26.4'; print(torch.cuda.get_device_name(0))"])
(OUT/'requirements-qualified.txt').write_text(subprocess.check_output([str(PY),'-m','pip','freeze'],env=env,text=True))
'''

def build(output, account, phase, source_kernel=None):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    slug=f"nano-gpu-{phase}-20261011-{uuid.uuid4().hex[:8]}"
    runner=Path(__file__).with_name("qualification_run.py").read_text(encoding="utf-8")
    execute=f"runner=OUT/'qualification_run.py'\nrunner.write_text({runner!r},encoding='utf-8')\ncommand([str(PY),str(runner),'--root',str(ROOT),'--output',str(OUT/'qualification'),'--phase',{phase!r}])\n"
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},"cells":[
        {"cell_type":"markdown","metadata":{},"source":[f"# {phase.title()} qualification — not research\nPinned main {COMMIT}. Uses shared nano_lm modules. Isolated approved dependency profile; no base-image replacement.\nNo pilot or ablation study."]},
        {"cell_type":"code","metadata":{},"source":BOOTSTRAP.replace('COMMIT_VALUE',COMMIT).splitlines(True),"outputs":[],"execution_count":None},
        {"cell_type":"code","metadata":{},"source":execute.splitlines(True),"outputs":[],"execution_count":None}]}
    (folder/'qualification.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    metadata={"id":f"{account}/{slug}","title":slug.replace('-',' ').title(),"code_file":"qualification.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":[],"kernel_sources":[source_kernel] if source_kernel else []}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'folder':str(folder),'kernel':metadata['id']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--account',required=True);p.add_argument('--phase',choices=['qualification','recovery'],default='qualification');p.add_argument('--source-kernel')
    a=p.parse_args();build(a.output,a.account,a.phase,a.source_kernel)
