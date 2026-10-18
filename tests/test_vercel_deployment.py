"""Keep deployment dependencies and routing separate from research."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_deployment_profile_and_runtime_entrypoint():
    profile = (ROOT / 'api/pyproject.toml').read_text()
    assert 'torch==2.5.1+cpu' in profile
    assert 'package=false' in profile
    assert 'torch={index="pytorch-cpu"}' in profile
    assert 'entrypoint="app:app"' in profile
    config = json.loads((ROOT / 'api/vercel.json').read_text())
    assert config['framework'] == 'fastapi'
    assert config['buildCommand'] == 'python vercel_build.py'
    assert config['functions']['app.py']['includeFiles'] == '_runtime/**'
    assert '/api/_runtime/' in (ROOT / '.gitignore').read_text()


def test_frontend_retains_backend_proxy_and_nextjs_profile():
    config = json.loads((ROOT / 'frontend/vercel.json').read_text())
    assert config['framework'] == 'nextjs'
    source = (ROOT / 'frontend/next.config.ts').read_text()
    assert 'NANOLM_BACKEND_ORIGIN' in source
    assert "source: '/api/:path*'" in source
    assert 'url.protocol' in source and 'url.password' in source


def test_platform_runtime_package_is_not_shadowed_by_api_source():
    # Vercel imports vercel_runtime.vc_init before our ASGI entrypoint.
    assert not (ROOT / 'api/vercel_runtime.py').exists()
    assert not (ROOT / 'api/vercel_runtime').exists()
    adapter = (ROOT / 'tools/prepare_vercel_backend.py').read_text()
    assert 'from api.bundled_runtime import create_hosted_app' in adapter
