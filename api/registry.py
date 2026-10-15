"""Locked lazy loading of locally verified safetensors; no pickle on serving path."""
import json
import threading
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from nano_lm.artifacts import file_hash
from nano_lm.bpe import from_artifact
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from .catalog import ROOT, catalog, TOKENIZER_HASH


class ModelUnavailable(Exception):
    """Publicly safe model availability error."""


class Registry:
    def __init__(self, artifacts: Path, device: str = 'cpu'):
        if device not in {'cpu', 'cuda'} or device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Requested inference device is unavailable')
        self.artifacts, self.device = Path(artifacts), device
        self.entries = catalog()
        self.models, self.load_seconds, self.errors = {}, {}, {}
        self.lock = threading.Lock()
        self.tokenizer = from_artifact(json.loads((ROOT / 'assets/tokenizers/bpe_canonical.json').read_text(encoding='utf-8')))
        if self.tokenizer.checksum != TOKENIZER_HASH:
            raise ValueError('Canonical tokenizer integrity failed')

    def load(self, model_id):
        if model_id not in self.entries:
            raise ModelUnavailable('Unknown model')
        with self.lock:
            if model_id in self.models:
                return self.models[model_id]
            start = time.perf_counter()
            try:
                directory = self.artifacts / model_id
                metadata_path = directory / 'metadata.json'
                if metadata_path.stat().st_size > 2**20:
                    raise ValueError('Oversized package metadata')
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                expected = self.entries[model_id]
                for key in ['id', 'architecture', 'parameters', 'checkpoint_step', 'tokenizer_hash',
                            'original_checkpoint_sha256', 'training_commit', 'validation_bpb']:
                    if metadata[key] != expected[key]:
                        raise ValueError('Identity mismatch')
                if metadata['format_version'] != 1 or metadata['precision'] != 'float32':
                    raise ValueError('Unsupported serving package')
                weights = directory / 'model.safetensors'
                size = weights.stat().st_size
                if size != metadata['artifact_bytes'] or not 0 < size <= expected['parameters'] * 4 + 2**20:
                    raise ValueError('Invalid inference artifact size')
                if weights.is_symlink() or file_hash(weights) != metadata['artifact_sha256']:
                    raise ValueError('Artifact integrity failed')
                state = load_file(str(weights), device='cpu')
                if any(t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all()) for t in state.values()):
                    raise ValueError('Invalid weight precision or values')
                model = GPT(ModelConfig(**expected['architecture']))
                model.load_state_dict(state, strict=True)
                if model.parameter_count() != expected['parameters']:
                    raise ValueError('Parameter mismatch')
                self.models[model_id] = model.to(self.device).eval()
                self.load_seconds[model_id] = time.perf_counter() - start
                self.errors.pop(model_id, None)
                return self.models[model_id]
            except Exception as exc:
                self.errors[model_id] = 'Verified inference artifact missing or invalid'
                raise ModelUnavailable(self.errors[model_id]) from exc

    def describe(self):
        # Presence is not verified readiness; no liveness request loads models.
        return [{**entry, 'availability': ('ready' if model_id in self.models else
                'unavailable' if model_id in self.errors or not (self.artifacts / model_id / 'metadata.json').is_file()
                else 'not_loaded')} for model_id, entry in self.entries.items()]
