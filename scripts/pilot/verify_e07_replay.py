"""Compare preserved original attempts with the recovered E07 trajectory, without updates."""
import argparse
import json
from pathlib import Path
from nano_lm.artifacts import atomic_json


def verify(evidence, output):
    original=json.loads((evidence/'original_saved_attempts.json').read_text())
    history=json.loads((evidence/'analysis/measured_history.json').read_text())['training']
    old={e['metric']['step']:e['metric'] for e in original['events'] if e['event']=='step'}
    new={row['step']:row for row in history}
    fields=['training_loss','gradient_norm','training_tokens','training_bytes','unique_target_bytes','amp_skipped_updates','lr','accumulation']
    steps=sorted(step for step in old if step>original['committed_resume_step'])
    differences=[{'step':step,'field':field,'old':old[step][field],'new':new[step][field]} for step in steps for field in fields if old[step][field]!=new[step][field]]
    assert not differences, 'Preserved attempt and resumed numerical trajectory differ'
    record={'durable_resume_step':original['committed_resume_step'],'original_saved_step_events':len(old),
            'last_original_saved_step':max(old),'replayed_saved_attempt_steps':steps,'replayed_step_count':len(steps),
            'numeric_fields_compared':fields,'exact_recorded_replay_match':True,'differences':[],
            'additional_optimizer_updates_for_this_check':0,
            'scope':'Actual original attempt metrics versus authoritative restored trajectory; elapsed times/memory excluded. No training recomputation.'}
    atomic_json(output/'replayed_attempt_comparison.json',record)
    extra=old[max(old)]['train_seconds']-new[original['committed_resume_step']]['train_seconds']
    count=new[max(new)]['training_tokens'];superseded=sum(old[step]['training_tokens']-old[step-1]['training_tokens'] for step in steps)
    timing={'authoritative_trajectory_training_seconds':new[max(new)]['train_seconds'],
            'known_superseded_attempt_training_seconds':extra,'known_minimum_physical_training_seconds':new[max(new)]['train_seconds']+extra,
            'authoritative_predicted_tokens':count,'known_superseded_attempt_tokens':superseded,'known_minimum_physical_predicted_tokens':count+superseded,
            'scope':'Genuine cumulative engine timings/counters. Unknown later volatile progress excluded; complete physical runtime is unavailable.'}
    atomic_json(output/'replay_timing.json',timing)
    print(f'{len(steps)} preserved replay steps match exactly; zero optimizer updates')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);verify(args.evidence,args.output)
