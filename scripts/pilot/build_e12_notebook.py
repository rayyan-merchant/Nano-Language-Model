"""Build only the authorized syedarijaali E12 notebook phases."""
import argparse
import json
from pathlib import Path


def build(commit, phase, output):
    if phase not in {'preflight', 'research', 'recover-final'}:
        raise ValueError('Unsupported E12 phase')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact committed source required')
    bootstrap=f'''from pathlib import Path
import os, sys, subprocess, json, time
ROOT=Path('/tmp/e12_source'); OUT=Path('/kaggle/working')
SOURCE={commit!r}
env={{**os.environ,'UV_PYTHON_INSTALL_DIR':'/tmp/e12_python','UV_CACHE_DIR':'/tmp/e12_cache','UV_NATIVE_TLS':'true','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/e12_mpl','MPLBACKEND':'Agg','NANO_SOURCE_COMMIT':SOURCE,'NANO_SESSION_LIMIT_SECONDS':'43200'}}
def command(args,cwd=None):
    print('EXEC',args,flush=True)
    with (OUT/'bootstrap.log').open('a') as log:
        result=subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print((OUT/'bootstrap.log').read_text()[-4000:],flush=True)
        raise RuntimeError('E12 gate failed; preserve output')
assert not ROOT.exists()
command(['git','clone','https://github.com/rayyan-merchant/Nano-Language-Model.git',str(ROOT)])
command(['git','checkout','--detach',SOURCE],ROOT)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
command([sys.executable,'-m','pip','install','--target','/tmp/e12_tools','uv==0.12.7'])
UV=Path('/tmp/e12_tools/bin/uv')
command([str(UV),'python','install','3.11.14'])
command([str(UV),'venv','--seed','--python','3.11.14','/tmp/e12_venv'])
PY=Path('/tmp/e12_venv/bin/python')
freeze=(ROOT/'docs/qualifications/evidence/rayyan_requirements_freeze.txt').read_text().splitlines()
lock=OUT/'qualified-lock.txt';lock.write_text(chr(10).join(x for x in freeze if not x.startswith('-e '))+chr(10))
command([str(UV),'pip','install','--python',str(PY),'torch==2.5.1+cu124','--index-url','https://download.pytorch.org/whl/cu124'])
command([str(UV),'pip','install','--python',str(PY),'-r',str(lock)])
command([str(UV),'pip','install','--python',str(PY),'-e',str(ROOT),'--no-deps'])
command([str(PY),'-m','pip','check'])
'''
    if phase=='preflight':
        bootstrap+="command([str(PY),str(ROOT/'scripts/pilot/e12_kaggle.py'),'preflight','--root',str(ROOT),'--output',str(OUT/'preflight')],ROOT)\n"
        bootstrap+="command([str(PY),str(ROOT/'scripts/pilot/e12_export_benchmark.py'),'--output',str(OUT)],ROOT)\n"
    elif phase=='research':
        bootstrap+='''archives=list(Path('/kaggle/input').rglob('E12_smoke_bundle.zip'))
assert len(archives)==1
archive=archives[0];manifest=archive.parent/'E12_smoke_manifest.json'
command([str(PY),str(ROOT/'scripts/pilot/e12_kaggle.py'),'recover-smoke','--root',str(ROOT),'--output',str(OUT/'persistent_recovery'),'--archive',str(archive),'--manifest',str(manifest)],ROOT)
subprocess.run([str(PY),str(ROOT/'scripts/pilot/e12_kaggle.py'),'research','--root',str(ROOT),'--output',str(OUT/'research'),'--store',str(OUT/'durable_staging'),'--recovery-receipt',str(OUT/'persistent_recovery/persistent_recovery.json')],cwd=ROOT,env=env,check=True)
command([str(PY),str(ROOT/'scripts/pilot/e12_kaggle.py'),'finish','--root',str(ROOT),'--output',str(OUT/'research'),'--store',str(OUT/'durable_staging')],ROOT)
'''
    else:
        bootstrap+="command([str(PY),str(ROOT/'scripts/pilot/recover.py'),'--root',str(ROOT),'--output',str(OUT/'fresh_final_recovery')],ROOT)\n"
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[{'id':'scope','cell_type':'markdown','metadata':{},'source':[f'# E12 seed42 {phase}\nOnly context256, accumulation4. Preflight/restore are development evidence. Research stops at5000. Source {commit}.']},{'id':'execute','cell_type':'code','metadata':{},'source':bootstrap.splitlines(True),'execution_count':None,'outputs':[]}]}
    Path(output).write_text(json.dumps(nb,indent=2),encoding='utf-8')
    return nb

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--phase',choices=['preflight','research','recover-final'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.commit,a.phase,a.output)
