"""Matched E01/E07 comparison from verified recorded checkpoints and committed E01."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nano_lm.artifacts import atomic_json, file_hash


def compare(root, evidence, output):
    output.mkdir(parents=True,exist_ok=False)
    baseline=root/'report/pilots/E01_seed42'
    a=json.loads((baseline/'summary.json').read_text())
    b=json.loads((evidence/'analysis/summary.json').read_text())
    h=json.loads((evidence/'analysis/measured_history.json').read_text())
    metadata=json.loads(next((evidence/'runs').glob('*/metadata.json')).read_text())
    assert a['parameters']==b['parameters']==1804800 and a['code_hash']==b['code_hash']
    assert a['completed_optimizer_updates']==b['completed_optimizer_updates']==5000
    baseline_config={'vocab_size':65,**json.loads((root/'configs/baseline.json').read_text())['model']}
    delta={k:v for k,v in metadata['resolved']['model'].items() if baseline_config[k]!=v}
    assert delta=={'n_heads':8}
    with (baseline/'validation.csv').open(newline='') as f:av=list(csv.DictReader(f))
    bv=h['validation'];assert [int(r['step']) for r in av]==[r['step'] for r in bv]==list(range(250,5001,250))
    rows=[]
    for x,y in zip(av,bv):
        loss_a=float(x['loss']);loss_b=y['validation']['loss'];bpb_a=float(x['bpb']);bpb_b=y['validation']['bpb']
        rows.append({'step':y['step'],'E01_loss':loss_a,'E07_loss':loss_b,'loss_difference_E07_minus_E01':loss_b-loss_a,
                     'E01_bpb':bpb_a,'E07_bpb':bpb_b,'bpb_difference_E07_minus_E01':bpb_b-bpb_a,
                     'relative_loss_reduction_percent':100*(loss_a-loss_b)/loss_a})
    with (output/'matched_validation.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures=output/'figures';figures.mkdir()
    for metric,ylabel in [('loss','Full sliding validation loss (nats/character)'),('bpb','Conditional bits per source byte')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for label in ['E01','E07']:ax.plot([r['step'] for r in rows],[r[label+'_'+metric] for r in rows],label=label+(' 6 heads' if label=='E01' else ' 8 heads'),marker='o',markersize=3)
        ax.set(xlabel='Completed optimizer updates',ylabel=ylabel,title='E01 vs E07 seed42: matched character protocol')
        ax.grid(alpha=.2);ax.legend();fig.text(.5,.01,'Single seed; equal parameters/context/exposure; separate Kaggle sessions; host/runtime confound',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(figures/f'e01_e07_validation_{metric}.png',dpi=200);fig.savefig(figures/f'e01_e07_validation_{metric}.pdf');plt.close(fig)
    gen_stats=[]
    for label,path in [('E01',baseline/'generation.json'),('E07',evidence/'analysis/generation.json')]:
        grid=json.loads(path.read_text());assert len(grid['samples'])==100
        for policy in ['best','final']:
            samples=[s for s in grid['samples'] if s['checkpoint_policy']==policy]
            for mode in ['greedy',.7,1.,1.2]:
                selected=[s for s in samples if s['greedy']] if mode=='greedy' else [s for s in samples if not s['greedy'] and s['temperature']==mode]
                fractions=[]
                for s in selected:
                    text=s['text'][len(s['prompt']):];grams=[text[i:i+8] for i in range(max(0,len(text)-7))]
                    fractions.append(1-len(set(grams))/len(grams) if grams else 0)
                gen_stats.append({'experiment':label,'policy':policy,'mode':mode,'samples':len(selected),
                                  'mean_repeated_8gram_position_fraction':float(np.mean(fractions)),
                                  'mean_continuation_characters':float(np.mean([len(s['text'])-len(s['prompt']) for s in selected]))})
    initial=json.loads((evidence/'heads_audit.json').read_text())
    norms=np.array([r['gradient_norm'] for r in h['training']])
    baseline_obs=json.loads((baseline/'pilot_observations.json').read_text())
    io=json.loads((evidence/'io_timing.json').read_text())
    receipt={'source_commit':b['source_commit'],'scientific_code_hash':b['code_hash'],'resolved_config_hash':b['config_hash'],
             'E01_reference_summary_sha256':file_hash(baseline/'summary.json'),'E01_reference_validation_sha256':file_hash(baseline/'validation.csv'),
             'initial_parameter_identity':initial['all_initial_E01_parameter_tensors_equal'],'sole_model_difference':delta,
             'matched_validation_points':20,'rows':rows,'E01':a,'E07':b,
             'best_loss_reduction_percent':100*(a['best_validation']['loss']-b['best_validation']['loss'])/a['best_validation']['loss'],
             'final_loss_reduction_percent':100*(a['final_validation']['loss']-b['final_validation']['loss'])/a['final_validation']['loss'],
             'training_time_change_percent':100*(b['training_seconds']-a['training_seconds'])/a['training_seconds'],
             'E01_clipping_frequency':baseline_obs['gradient_clipping_fraction'],'E07_clipping_frequency':float((norms>1).mean()),
             'checkpoint_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='checkpoint'),
             'snapshot_export_seconds':sum(r['seconds'] for r in io['records'] if r['operation']=='export'),
             'journal_seconds':io['journal_seconds'],'engine_wall_seconds':io['engine_wall_seconds'],'io_scope':io.get('scope','complete original research segment'),
             'generation_repetition':gen_stats,'limit':'One seed; no statistical significance. Separate host/session scheduling and runtime conditions uncontrolled. Repetition is descriptive, not a quality score.'}
    atomic_json(output/'comparison.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.root,a.evidence,a.output)
