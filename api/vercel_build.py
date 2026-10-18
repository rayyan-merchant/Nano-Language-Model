"""Stage canonical source and hash-verified public inference weights for Vercel."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent / '_runtime'


def build():
    sys.path.insert(0, str(ROOT))
    from tools.submission_models import download, checksums
    DEST.mkdir(exist_ok=True)
    for name in ['src', 'configs', 'experiments', 'assets', 'report']:
        shutil.copytree(ROOT / name, DEST / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    target = DEST / 'api'
    target.mkdir(exist_ok=True)
    for source in (ROOT / 'api').glob('*.py'):
        if source.name not in {'app.py', 'vercel_build.py'}:
            shutil.copyfile(source, target / source.name)
    models = DEST / 'model_artifacts'
    if models.exists():
        checksums(models)
    else:
        download(models)
    print('Staged unchanged scientific source and three hash-verified model packages.')


if __name__ == '__main__':
    build()
