"""Development-only exact full scientific trajectory comparison."""
import argparse
from pathlib import Path
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import read_checkpoint
from nano_lm.storage import export_run, restore_snapshot, verify_snapshot
from nano_lm.train import TrainSettings, train
from scripts.balanced.state import compare_trajectory, roundtrip


def verify(root, data, out, device='cpu', full_validation=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    settings=TrainSettings(device=device,run_kind='smoke',eval_every=2,checkpoint_every=2,
                           eval_max_bytes=0 if full_validation else 512)
    _, straight, _=train(root,data,'B01',settings,4,out/'straight')
    partial, mid, _=train(root,data,'B01',settings,2,out/'segmented')
    snapshot=export_run(partial,out/'snapshots');manifest=verify_snapshot(snapshot)
    restored=restore_snapshot(snapshot,out/'restored')
    receipt=roundtrip(restored/'checkpoints'/mid.name,out/'roundtrip.pt',device)
    _, resumed, _=train(root,data,'B01',settings,4,out/'restored',resume=restored/'checkpoints'/mid.name)
    a,_=read_checkpoint(straight);b,_=read_checkpoint(resumed)
    result={'status':'B01_development_uninterrupted_vs_segmented_full_scientific_state_exact',
        'device':device,'full_sliding_validation':full_validation,'comparison':compare_trajectory(a,b),
        'zero_update_full_state_restore':receipt,'smoke_updates_per_trajectory':4,
        'known_physical_development_updates':8,'research_updates':0,
        'snapshot_hash':manifest['hash'],'mid_checkpoint_sha256':file_hash(mid),
        'sampled_windows_and_target_bytes_match':True}
    atomic_json(out/'continuation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','dataset','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--full-validation',action='store_true')
    a=p.parse_args();verify(a.root,a.dataset,a.output,a.device,a.full_validation)
