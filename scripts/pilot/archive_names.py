"""Resolve explicit published archive names without evaluating legacy fallback."""
from pathlib import Path


def archive_name(reference):
    if 'archive_file' in reference:
        name=reference['archive_file']
    else:
        name=reference['run_id'].split('_seed42_')[0]+'_seed42_pilot.zip'
    if not isinstance(name,str) or not name or Path(name).name!=name or '/' in name or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive filename')
    return name
