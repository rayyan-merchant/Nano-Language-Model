"""Three-way matched head-count comparison; E06 must come from reviewed main."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nano_lm.artifacts import atomic_json, digest


def compare(root, e07, output, reviewed_ref):
    # Git provides the exact reviewed reference bytes without editing E06 evidence.
    read = lambda path: subprocess.check_output(['git','show',f'{reviewed_ref}:{path}'],cwd=root).decode()
    prefix='report/pilots/E06_seed42/'
    e06=json.loads(read(prefix+'summary.json'))
    a=json.loads((root/'report/pilots/E01_seed42/summary.json').read_text())
    b=json.loads((e07/'analysis/summary.json').read_text())
    refs={'E06':e06,'E01':a,'E07':b}
    assert all(s['parameters']==1804800 and s['completed_optimizer_updates']==5000 and s['code_hash']==a['code_hash'] for s in refs.values())
    assert json.loads(read('experiments/configs/E06.json'))['model_overrides']=={'n_heads':4}
    curves={'E06':list(csv.DictReader(read(prefix+'validation.csv').splitlines()))}
    for name,path in [('E01',root/'report/pilots/E01_seed42/validation.csv'),('E07',e07/'analysis/validation.csv')]:
        curves[name]=list(csv.DictReader(path.read_text().splitlines()))
    assert all([int(row['step']) for row in curve]==list(range(250,5001,250)) for curve in curves.values())
    rows=[{'step':int(curves['E01'][i]['step']),**{f'{name}_{metric}':float(curve[i][metric]) for name,curve in curves.items() for metric in ['loss','bpb']}} for i in range(20)]
    output.mkdir(parents=True,exist_ok=False)
    with (output/'matched_three_heads.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels={'E06':'4 heads / 48 dimensions','E01':'6 heads / 32 dimensions','E07':'8 heads / 24 dimensions'}
    for metric in ['loss','bpb']:
        fig,ax=plt.subplots(figsize=(8,4.8))
        for name in refs:ax.plot([r['step'] for r in rows],[r[f'{name}_{metric}'] for r in rows],label=name+' '+labels[name],marker='o',markersize=3)
        ax.set(xlabel='Completed optimizer updates',ylabel='Validation loss (nats/character)' if metric=='loss' else 'Conditional bits per byte',title='Seed 42: fixed width 192, 1,804,800 parameters')
        ax.legend();ax.grid(alpha=.2)
        fig.text(.5,.015,'One seed; separate Kaggle sessions; timing and attention mechanisms not isolated',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.045,1,1])
        for suffix in ['png','pdf']:fig.savefig(output/f'three_heads_{metric}.{suffix}',dpi=200)
        plt.close(fig)
    atomic_json(output/'three_heads.json',{'reviewed_reference_commit':reviewed_ref,'E06_reference_summary_digest':digest(e06),'runs':refs,'rows':rows,'scope':'Approved merged E06 report, unchanged E01 reference and independently restored E07. No E06 training or edits performed.'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','e07','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewed-ref',required=True)
    a=p.parse_args();compare(a.root,a.e07,a.output,a.reviewed_ref)
