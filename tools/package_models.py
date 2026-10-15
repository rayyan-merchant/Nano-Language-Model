"""Hash-anchored trusted checkpoint conversion; never exposed through the API."""
import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from nano_lm.artifacts import file_hash
from nano_lm.checkpoint import read_checkpoint
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.generate import generate
from api.catalog import catalog, TOKENIZER_HASH


def export(model_id: str, checkpoint: Path, destination: Path):
    selected = catalog()[model_id]
    if file_hash(checkpoint) != selected['original_checkpoint_sha256']:
        raise ValueError('Original checkpoint SHA256 does not match accepted research evidence')
    if destination.exists():
        raise FileExistsError('Use a new immutable package destination')
    payload, tokenizer = read_checkpoint(checkpoint)  # trusted, hash-verified local pickle only
    if payload['step'] != selected['checkpoint_step'] or payload['config'] != selected['architecture']:
        raise ValueError('Wrong selected update or architecture')
    metadata = payload['metadata']
    if metadata['resolved']['experiment_id'] != model_id.upper() or tokenizer.checksum != TOKENIZER_HASH:
        raise ValueError('Wrong experiment or canonical tokenizer')
    source = {key: metadata['environment'][key] for key in ['git_commit', 'code_hash', 'git_dirty']}
    if source['git_commit'] != selected['training_commit'] or source['git_dirty']:
        raise ValueError('Wrong original training commit')
    original = GPT(ModelConfig(**payload['config'])).eval()
    original.load_state_dict(payload['model'], strict=True)
    if original.parameter_count() != selected['parameters']:
        raise ValueError('Wrong parameter count')
    state = {k: v.detach().cpu().contiguous().clone() for k, v in original.state_dict().items()}
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Weights must be finite canonical float32')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.package-', dir=destination.parent))
    try:
        save_file(state, str(staging / 'model.safetensors'))
        restored_state = load_file(str(staging / 'model.safetensors'))
        assert set(restored_state) == set(state)
        assert all(torch.equal(v, restored_state[k]) and v.dtype == restored_state[k].dtype for k, v in state.items())
        restored = GPT(ModelConfig(**payload['config'])).eval()
        restored.load_state_dict(restored_state, strict=True)
        receipts = []
        for prompt in ['ROMEO:\n', 'To be, or not to be', 'é ' + 'The king ' * 60]:
            ids = torch.tensor([tokenizer.encode(prompt)])
            with torch.no_grad():
                assert torch.equal(original(ids[:, -128:])[0], restored(ids[:, -128:])[0])
            for greedy in [True, False]:
                before = generate(original, ids, 32, 0.7, greedy, 42)
                after = generate(restored, ids, 32, 0.7, greedy, 42)
                assert torch.equal(before, after)
                receipts.append({'prompt': prompt, 'greedy': greedy, 'seed': 42, 'new_tokens': 32,
                                 'token_ids': after[0, ids.shape[1]:].tolist(),
                                 'decoded': tokenizer.decode(after[0].tolist()), 'exact_ids_equal': True})
        manifest = {**selected, 'format_version': 1, 'precision': 'float32',
                    'artifact_sha256': file_hash(staging / 'model.safetensors'),
                    'artifact_bytes': (staging / 'model.safetensors').stat().st_size,
                    'dataset_hash': metadata['resolved']['dataset_hash'],
                    'configuration_hash': metadata['config_hash'], 'scientific_source': source,
                    'export_environment': {'python': platform.python_version(), 'torch': torch.__version__},
                    'equivalence': {'tensor_values_and_dtypes': True, 'logits_exact': True,
                                    'cases': receipts, 'device': 'cpu', 'optimizer_updates': 0}}
        (staging / 'metadata.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(catalog()), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New model directory, e.g. model_artifacts/b01')
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    receipt = export(args.model, args.checkpoint, args.output)
    print(json.dumps({'id': args.model, 'bytes': receipt['artifact_bytes'], 'sha256': receipt['artifact_sha256'],
                      'exact_equivalence': True, 'seconds': time.perf_counter() - start}))


if __name__ == '__main__':
    main()
