"""Acquire canonical source and reuse audited tokenizers, never fit vocabulary."""
import hashlib
import json
from pathlib import Path
from nano_lm.artifacts import atomic_bytes, atomic_json
from nano_lm.prepare import acquire, validate, EXPECTED_SHA256
from nano_lm.data import split_text
from nano_lm.bpe import from_artifact
from scripts.balanced.protocol import BPE


def prepare(root, output):
    root, output=Path(root),Path(output)
    reference=json.loads((root/'docs/evidence/dataset_manifest.json').read_text(encoding='utf-8'))
    if output.exists():
        actual=validate(output);assert actual==reference;return actual
    output.mkdir(parents=True,exist_ok=False)
    raw=acquire('http');assert hashlib.sha256(raw).hexdigest()==EXPECTED_SHA256
    parts,boundaries=split_text(raw.decode('utf-8'));assert boundaries==reference['split']
    atomic_bytes(output/'source.txt',raw)
    for name,text in parts.items():atomic_bytes(output/f'{name}.txt',text.encode('utf-8'))
    for kind in ['character','bpe']:
        artifact=(root/f'assets/tokenizers/{kind}_canonical.json').read_bytes()
        tokenizer=from_artifact(json.loads(artifact))
        assert tokenizer.checksum==reference['tokenizers'][kind]['hash']
        if kind=='bpe':assert tokenizer.checksum==BPE and tokenizer.vocab_size==1024
        atomic_bytes(output/f'{kind}.json',artifact)
    atomic_json(output/'manifest.json',reference)
    return validate(output)
