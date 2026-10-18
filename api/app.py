"""Thin Vercel entrypoint; the scientific implementation is copied unchanged."""
import os
from pathlib import Path
import sys

runtime = Path(__file__).resolve().parent / '_runtime'
sys.path.insert(0, str(runtime))
sys.path.insert(0, str(runtime / 'src'))
os.environ['NANOLM_ARTIFACTS'] = str(runtime / 'model_artifacts')
from api.main import app
