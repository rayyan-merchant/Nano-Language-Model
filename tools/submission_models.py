"""Anonymous, revision-pinned retrieval of the three approved inference packages.

No Hub token, user cache, pickle loader, URL override or local machine path is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/submission/models.json'


def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def checksums(directory):
    """Validate pinned file bytes and metadata before any tensor loader is reached."""
    records = []
    for model_id, expected in manifest()['models'].items():
        for filename, prefix in [('metadata.json', 'metadata'), ('model.safetensors', 'artifact')]:
            path = Path(directory) / model_id / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Missing or unsafe package file: {model_id}/{filename}')
            if path.stat().st_size != expected[prefix + '_bytes']:
                raise ValueError(f'Package size mismatch: {model_id}/{filename}')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected[prefix + '_sha256']:
                raise ValueError(f'Package checksum mismatch: {model_id}/{filename}')
            records.append({'path': f'{model_id}/{filename}', 'bytes': path.stat().st_size, 'sha256': actual})
        metadata = json.loads((Path(directory) / model_id / 'metadata.json').read_text(encoding='utf-8'))
        for key in ['id', 'parameters', 'checkpoint_step', 'architecture', 'tokenizer_hash',
                    'original_checkpoint_sha256', 'training_commit', 'precision', 'artifact_sha256']:
            if metadata[key] != expected[key]:
                raise ValueError(f'Selected model identity mismatch: {model_id}/{key}')
    return records


def download(destination, *, opener=urllib.request.urlopen):
    """Stage a complete release, reject partial files and publish without replacement."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Use a new package directory; existing evidence is never overwritten')
    destination.parent.mkdir(parents=True, exist_ok=True)
    release = manifest()
    staging = Path(tempfile.mkdtemp(prefix='.nanolm-download-', dir=destination.parent))
    try:
        for model_id, spec in release['models'].items():
            (staging / model_id).mkdir()
            for filename, prefix in [('metadata.json', 'metadata'), ('model.safetensors', 'artifact')]:
                url = f"https://huggingface.co/{release['repository']}/resolve/{release['revision']}/{model_id}/{filename}"
                request = urllib.request.Request(url, headers={'User-Agent': 'NanoLM-Submission/1'})
                try:
                    with opener(request, timeout=90) as response, (staging / model_id / filename).open('xb') as stream:
                        total = 0
                        while chunk := response.read(1024 * 1024):
                            total += len(chunk)
                            if total > spec[prefix + '_bytes']:
                                raise ValueError(f'Oversized download: {model_id}/{filename}')
                            stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(f'Model download HTTP{exc.code}; check the documented public repository access') from None
        files = checksums(staging)
        receipt = {'repository': release['repository'], 'revision': release['revision'],
                   'authentication': 'none; anonymous HTTPS, no Hub cache', 'files': files,
                   'all_checks_passed': True}
        (staging / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        # Reserve the final directory exclusively. Competing writers cannot replace evidence.
        destination.mkdir()
        try:
            for model_id in release['models']:
                (staging / model_id).rename(destination / model_id)
            # Completion marker is published last; failed publication never has a receipt.
            (staging / 'download_receipt.json').rename(destination / 'download_receipt.json')
        except BaseException:
            # A partially published directory remains clearly incomplete, never accepted silently.
            raise
        return receipt
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = checksums(args.output) if args.verify_only else download(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
