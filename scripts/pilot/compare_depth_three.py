"""Validate and compare genuine E01/E08/E09 seed42 checkpoints; no inference."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
from nano_lm.artifacts import atomic_json
from nano_lm.checkpoint import read_checkpoint
from nano_lm.results import collect


def compare(artifacts, root, output):
    output.mkdir(parents=True, exist_ok=False)
    ids = ['E01', 'E08', 'E09']; layers = [4, 6, 8]
    parameters = [1804800, 2694528, 3584256]
    payloads, summaries, runs = [], [], []
    for eid, count, size, folder in zip(ids, layers, parameters, artifacts):
        c = json.loads((folder/'completion.json').read_text())
        checkpoint = folder/c['final_checkpoint']; p, _ = read_checkpoint(checkpoint)
        assert p['step'] == 5000 and p['metadata']['resolved']['experiment_id'] == eid
        assert p['config']['n_layers'] == count and p['metadata']['parameters'] == size
        payloads.append(p); runs.append(checkpoint.parent.parent)
        s = json.loads((folder/'analysis/summary.json').read_text())
        assert s['run_id'] == c['run_id'] and s['completed_optimizer_updates'] == 5000
        summaries.append(s)
    verified, aggregate = collect(runs, root, 5000, require_complete=False)
    first = payloads[0]; a = first['metadata']
    for p in payloads[1:]:
        b = p['metadata']
        assert a['resolved']['training'] == b['resolved']['training']
        assert a['resolved']['tokenizer_hash'] == b['resolved']['tokenizer_hash']
        assert a['resolved']['tokenizer_kind'] == b['resolved']['tokenizer_kind'] == 'character'
        assert {k for k in first['config'] if first['config'][k] != p['config'][k]} == {'n_layers'}
        freeze = lambda e: sorted(x for x in e['packages'] if not x.startswith('-e '))
        assert freeze(a['environment']) == freeze(b['environment'])
        assert all(a['environment'][k] == b['environment'][k] for k in ['code_hash','python','torch','cuda','gpu'])
        assert torch.equal(first['rng']['sampling'], p['rng']['sampling'])
        assert np.array_equal(first['training_state']['seen_bytes'], p['training_state']['seen_bytes'])
    histories = [p['training_state']['history'] for p in payloads]
    values = [[r for r in h if 'validation' in r] for h in histories]
    assert all([r['step'] for r in v] == list(range(250,5001,250)) for v in values)
    rows = []
    for records in zip(*values):
        row = {'step': records[0]['step']}
        for eid, r in zip(ids, records):
            assert r['validation']['scored_bytes'] == 55514 and r['validation']['conditioning_bytes'] == 256
            row.update({f'{eid}_{m}': r['validation'][m] for m in ['loss','bpb','perplexity']})
            row[f'{eid}_training_seconds'] = r['train_seconds']
        rows.append(row)
    pairs = []
    for left, right in [(0,1),(0,2),(1,2)]:
        for metric, key, sub in [('best_loss','best_validation','loss'),('final_loss','final_validation','loss'),('best_bpb','best_validation','bpb'),('final_bpb','final_validation','bpb'),('training_seconds','training_seconds',None),('validation_seconds','validation_seconds',None),('parameters','parameters',None),('peak_gpu_bytes','peak_gpu_bytes',None)]:
            x,y = [s[key][sub] if sub else s[key] for s in [summaries[left],summaries[right]]]
            pairs.append({'reference':ids[left],'candidate':ids[right],'metric':metric,'reference_value':x,'candidate_value':y,'delta':y-x,'relative_percent':100*(y-x)/x})
    gradients = []
    for eid, h in zip(ids, histories):
        norms = np.array([r['gradient_norm'] for r in h]); assert np.isfinite(norms).all()
        gradients.append({'experiment':eid,'maximum':float(norms.max()),'median':float(np.median(norms)),'p95':float(np.percentile(norms,95)),'clipped_updates':int((norms>1).sum()),'first100_mean':float(norms[:100].mean()),'last100_mean':float(norms[-100:].mean()),'scope':'Pre-clipping aggregate norms; no longitudinal per-layer gradient measurements'})
    atomic_json(output/'comparison.json',{'schema_version':1,'label':'E01/E08/E09 seed42 measured depth/capacity comparison','comparability_passed':True,'only_model_factor_difference':'n_layers','same_sampling_state_and_byte_coverage':True,'summaries':dict(zip(ids,summaries)),'matched_validation':rows,'pairwise_differences':pairs,'gradients':gradients,'verified_runs':verified,'aggregate':aggregate,'limitations':['One seed per configuration; SD undefined, no significance claim','Depth, capacity, initialization/RNG consumption and computation change together','Same GPU model and freeze, separate sessions; timings not controlled repetitions','No test likelihood/selection; character BPB and NLL are related']})
    for name, data in [('matched_validation',rows),('pairwise_differences',pairs),('gradients',gradients),('verified_runs',verified),('aggregate',aggregate)]:
        with (output/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def lines(name, metric, time_axis=False):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for eid,depth,v in zip(ids,layers,values):
            x=[r['train_seconds'] if time_axis else r['step'] for r in v]
            ax.plot(x,[r['validation'][metric] for r in v],marker='.',label=f'{eid}: {depth} blocks')
        ax.set(xlabel='Measured cumulative training-only seconds' if time_axis else 'Completed optimizer updates',ylabel='Conditional validation bits/byte' if metric=='bpb' else 'Full sliding validation NLL (nats)',title='Measured single-seed depth/capacity ablation')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=200);plt.close(fig)
    lines('validation_loss','loss');lines('validation_bpb','bpb');lines('validation_bpb_training_time','bpb',True)
    for name,key,scale,label in [('bpb_parameters','parameters',1e6,'Trainable parameters (millions)'),('bpb_memory','peak_gpu_bytes',2**20,'Measured peak allocated GPU memory (MiB)'),('bpb_training_cost','training_seconds',1,'Measured full5000-update training-only seconds')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for eid,s in zip(ids,summaries):
            x=s[key]/scale;y=s['best_validation']['bpb'];ax.scatter(x,y);ax.annotate(eid,(x,y),xytext=(5,5),textcoords='offset points')
        ax.set(xlabel=label,ylabel='Best observed conditional validation BPB',title='Three genuine seed42 observations; no fitted trend')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=200);plt.close(fig)
    for name,key,label in [('training_seconds','training_seconds','Measured training-only seconds'),('validation_seconds','validation_seconds','Measured20 full-validation seconds'),('memory','peak_gpu_bytes','Peak allocated GPU memory (MiB)')]:
        fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(ids,[s[key]/(2**20 if key=='peak_gpu_bytes' else 1) for s in summaries])
        ax.set(ylabel=label,title='Single-seed runs; separate TeslaT4 sessions');fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=200);plt.close(fig)
    print(json.dumps({'status':'three_way_comparability_passed','validation_points':20,'figures':9}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['baseline','six-layer','eight-layer','root','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare([a.baseline,a.six_layer,a.eight_layer],a.root,a.output)
