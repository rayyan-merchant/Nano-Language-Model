"""Capture saved Kaggle logs/timestamps with the separately installed Kaggle CLI API."""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def capture(kernel,out):
    if not kernel.startswith('rayyannmerchant/'):
        raise ValueError('Only the authorized account is permitted')
    api=KaggleApi();api.authenticate()
    status=api.kernels_status(kernel)
    if not any(x in str(status.status) for x in ['COMPLETE','ERROR']):raise RuntimeError('Capture only after terminal saved-output status')
    events=list(api.kernels_logs_stream(kernel));out.mkdir(parents=True,exist_ok=False)
    (out/'platform_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    def matching(text):
        return [float(e['time']) for e in events if 'time' in e and text in e.get('data','')]
    starts=matching("EXEC ['git', 'clone'");ends=matching('PILOT COMPLETE ARCHIVE')+matching('PILOT RECOVERY COMPLETE ARCHIVE')
    if not ends:ends=matching('Pilot failed; preserve outputs and inspect failure evidence')
    times=[float(e['time']) for e in events if 'time' in e]
    summary={'kernel':kernel,'status':str(status.status),'first_recorded_log_seconds':min(times),'last_recorded_log_seconds':max(times),'execution_pipeline_start_seconds':min(starts) if starts else None,'execution_pipeline_complete_seconds':max(ends) if ends else None,'pipeline_wall_seconds':max(ends)-min(starts) if starts and ends else None,'scope':'Platform-measured elapsed timestamps: clone/environment setup through training, analysis, archive hash completion. Excludes queue before execution, final notebook conversion/publication and network retrieval. Log span is not exact platform session lifetime.'}
    (out/'platform_timing.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kernel');p.add_argument('output',type=Path)
    a=p.parse_args();capture(a.kernel,a.output)
