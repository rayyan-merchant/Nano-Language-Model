"""Describe genuine learned-position table changes; no causal/quality inference."""
import argparse
import json
import os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from nano_lm.artifacts import atomic_json,file_hash
from nano_lm.checkpoint import read_checkpoint
from nano_lm.config import ModelConfig
from nano_lm.model import GPT


def diagnose(artifacts,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    completion=json.loads((artifacts/'completion.json').read_text())
    payload,_=read_checkpoint(artifacts/completion['final_checkpoint'])
    assert payload['config']['position']=='learned' and payload['step']==5000
    torch.manual_seed(42);initial=GPT(ModelConfig(**payload['config'])).position_embedding.weight.detach()
    rows=[]
    for policy,key in [('best','best_checkpoint'),('final','final_checkpoint')]:
        checkpoint=artifacts/completion[key];p,_=read_checkpoint(checkpoint)
        table=p['model']['position_embedding.weight'].cpu();delta=table-initial
        assert table.shape==(128,192) and torch.isfinite(table).all()
        norm=table.norm(dim=1);change=delta.norm(dim=1)
        rows.append({'policy':policy,'step':p['step'],'checkpoint_sha256':file_hash(checkpoint),
                     'row_norms':norm.tolist(),'row_change_norms_from_canonical_seed42_initialization':change.tolist(),
                     'mean_row_norm':norm.mean().item(),'mean_row_change_norm':change.mean().item(),
                     'all_position_rows_changed':bool(torch.all(change>0))})
    atomic_json(out/'learned_table.json',{'status':'finite128x192_table_measured','rows':rows,
                'initialization':'Unmodified canonical GPT construction, CPU torch.manual_seed42; matches float32 CPU initialization used before CUDA transfer',
                'limit':'Descriptive parameter changes only, not proof of learned syntax, useful position specialization or causal performance impact'})
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(initial.norm(dim=1),label='Canonical seed42 initial')
    for row in rows:ax.plot(row['row_norms'],label=f"{row['policy']} update{row['step']}")
    ax.set(xlabel='Local context position index',ylabel='Position-vector L2 norm',title='E04 seed42: genuine learned-position table norms')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'position_norms.png',dpi=200);plt.close(fig)
    print(json.dumps([{k:v for k,v in r.items() if 'norms' not in k} for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnose(a.artifacts,a.output)
