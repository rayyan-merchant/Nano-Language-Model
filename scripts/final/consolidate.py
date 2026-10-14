"""Read all historical reports and recorded summaries without modifying evidence."""
import argparse
import csv
import json
from pathlib import Path
from nano_lm.artifacts import atomic_json, file_hash
from scripts.final.protocol import provenance, configuration, parameter_breakdown


def consolidate(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(1, 15):
        eid = f'E{index:02d}'
        folder = root/'report/pilots'/f'{eid}_seed42'
        report = folder/f'{eid}_seed42_analysis.md'
        text = report.read_text(encoding='utf-8')
        summary_path = folder/'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        config = json.loads((root/f'experiments/configs/{eid}.json').read_text())
        generations = folder/'generation.json'
        # Preserve schema diversity instead of guessing missing historical fields.
        record = dict(experiment=eid, config=config, summary=summary,
                      report_path=report.relative_to(root).as_posix(), report_sha256=file_hash(report),
                      report_characters_read=len(text), summary_sha256=file_hash(summary_path),
                      interpretation_paths=[p.relative_to(root).as_posix() for p in folder.glob('*interpretation*')],
                      generation_path=generations.relative_to(root).as_posix() if generations.exists() else None)
        if generations.exists():
            grid = json.loads(generations.read_text(encoding='utf-8'))
            record['generation_sha256'] = file_hash(generations)
            record['generation_slots'] = len(grid.get('samples', []))
        records.append(record)
    evidence = {'status':'historical evidence inventory; final F01 untrained',
                'source_provenance':provenance(root), 'F01_model':configuration(root).to_dict(),
                'F01_parameter_derivation':parameter_breakdown(), 'pilots':records}
    atomic_json(out/'pilot_evidence.json', evidence)
    print(json.dumps([{'experiment':r['experiment'], 'summary_keys':list(r['summary']),
                       'best':r['summary'].get('best_validation'), 'generation_slots':r.get('generation_slots')}
                      for r in records], indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();consolidate(a.root,a.output)
