"""Vercel CPU adapter with hash-verified PRIVATE server-function model files.

No model directory is mounted as HTTP static content. Original scientific code,
weights, tokenizer and generation are reused without numerical transformations.
"""
import threading
from pathlib import Path
import json
from .hosting import MANIFEST, verify
from .registry import Registry, ModelUnavailable
from .catalog import ROOT


class BundledRegistry(Registry):
    def __init__(self, directory):
        self.package_lock, self.verified = threading.Lock(), False
        super().__init__(Path(directory), 'cpu')

    def load(self, model_id):
        try:
            with self.package_lock:
                if not self.verified:
                    verify(self.artifacts, json.loads(MANIFEST.read_text(encoding='utf-8')))
                    self.verified = True
        except Exception:
            raise ModelUnavailable('Verified private model package unavailable') from None
        return super().load(model_id)


def create_hosted_app():
    from .main import create_app
    return create_app(BundledRegistry(ROOT/'private_models'))
