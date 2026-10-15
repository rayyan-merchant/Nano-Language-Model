"""Local NanoLM Studio API with bounded inference and truthful availability."""
import asyncio
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .catalog import ROOT
from .inference import run
from .registry import Registry, ModelUnavailable
from .schemas import GenerateRequest, Settings, GenerationResult, CompareResult, ModelsResponse

LOG = logging.getLogger('nanolm.studio')


def create_app(registry=None, timeout=None, rate_limit=None):
    timeout = float(timeout or os.getenv('NANOLM_TIMEOUT_SECONDS', '60'))
    rate_limit = int(rate_limit or os.getenv('NANOLM_REQUESTS_PER_MINUTE', '20'))
    if timeout <= 0 or rate_limit < 1:
        raise ValueError('Positive service limits required')
    torch.set_num_threads(int(os.getenv('NANOLM_CPU_THREADS', '1')))
    registry = registry or Registry(Path(os.getenv('NANOLM_ARTIFACTS', str(ROOT / 'model_artifacts'))),
                                    os.getenv('NANOLM_DEVICE', 'cpu'))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nanolm')
    slot = threading.Lock()
    clients = OrderedDict()

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='NanoLM Studio', version='1.0.0', lifespan=lifespan)
    app.state.registry = registry

    def error(code, message, status, request_id=None):
        return JSONResponse(status_code=status, content={'error': {'code': code, 'message': message},
                                                        'request_id': request_id or uuid.uuid4().hex})

    @app.middleware('http')
    async def boundaries(request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        if request.method == 'POST':
            address = request.client.host if request.client else 'unknown'
            now = time.monotonic()
            history = clients.setdefault(address, deque())
            clients.move_to_end(address)
            while history and now-history[0] >= 60:
                history.popleft()
            if len(history) >= rate_limit:
                response = error('rate_limit', 'Too many requests. Try again in a minute.', 429, request_id)
                response.headers['Retry-After'] = '60'
                return response
            history.append(now)
            if len(clients) > 10000:
                clients.popitem(last=False)
            if request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return error('content_type', 'Use application/json.', 415, request_id)
            # Stream and bound the actual body, including chunked requests. Do not
            # trust Content-Length and do not retain prompts in logs.
            async def read_body():
                chunks, size = [], 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 32768:
                        return None
                    chunks.append(chunk)
                return b''.join(chunks)
            try:
                body = await asyncio.wait_for(read_body(), timeout=10)
            except asyncio.TimeoutError:
                return error('body_timeout', 'Request body took too long to arrive.', 408, request_id)
            if body is None:
                return error('request_size', 'Request body exceeds 32 KiB.', 413, request_id)
            request._body = body
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def validation(request, exc):
        return error('validation', 'Check the prompt, model and generation settings.', 422, request.state.request_id)

    @app.exception_handler(Exception)
    async def internal(request, exc):
        LOG.error('request=%s failed type=%s', request.state.request_id, type(exc).__name__)
        return error('internal', 'Inference failed. Please retry.', 500, request.state.request_id)

    async def execute(request, function):
        if not slot.acquire(blocking=False):
            return error('busy', 'The inference worker is busy. Please retry shortly.', 429, request.state.request_id)
        loop = asyncio.get_running_loop()
        def work():
            try:
                return function()
            finally:
                slot.release()
        future = loop.run_in_executor(executor, work)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            # Canonical generation has no cancellation hook. Keep the slot occupied
            # until the bounded running operation finishes; never enqueue extra work.
            return error('timeout', 'Generation timed out; the worker is finishing safely.', 504, request.state.request_id)
        except ModelUnavailable:
            return error('model_unavailable', 'Selected verified model artifact is unavailable.', 503, request.state.request_id)

    @app.get('/health')
    def health():
        return {'status': 'alive', 'service': 'NanoLM Studio', 'streaming': False}

    @app.get('/ready')
    async def ready(request: Request):
        def verify_models():
            for model_id in registry.entries:
                try:
                    registry.load(model_id)
                except ModelUnavailable:
                    pass
            models = registry.describe()
            ok = all(m['availability'] == 'ready' for m in models)
            return JSONResponse(status_code=200 if ok else 503, content={'ready': ok, 'models': models})
        return await execute(request, verify_models)

    @app.get('/v1/models', response_model=ModelsResponse)
    def models():
        return {'models': registry.describe(), 'loading_policy': 'lazy_locked', 'device': registry.device}

    @app.post('/v1/generate', response_model=GenerationResult, responses={503: {'description': 'Model unavailable'}, 429: {'description': 'Busy or rate limited'}, 504: {'description': 'Timed out'}})
    async def generation(body: GenerateRequest, request: Request):
        return await execute(request, lambda: run(registry, body, request.state.request_id))

    @app.post('/v1/compare', response_model=CompareResult)
    async def compare(body: Settings, request: Request):
        seed = body.seed if body.seed is not None else secrets.randbelow(2147483648)
        def compare_all():
            start, results = time.perf_counter(), []
            for model_id in registry.entries:
                try:
                    result = run(registry, GenerateRequest(**body.model_dump(), model_id=model_id), request.state.request_id, seed)
                    results.append({'model_id': model_id, 'status': 'completed', 'result': result.model_dump()})
                except ModelUnavailable:
                    results.append({'model_id': model_id, 'status': 'unavailable', 'error': 'Verified model artifact unavailable'})
                except Exception:
                    results.append({'model_id': model_id, 'status': 'failed', 'error': 'Model inference failed'})
            return CompareResult(request_id=request.state.request_id, seed=seed, results=results,
                                 total_latency_seconds=time.perf_counter()-start)
        return await execute(request, compare_all)

    # CORS must wrap early body/rate-limit errors as well as route responses.
    app.add_middleware(CORSMiddleware,
        allow_origins=[x.strip() for x in os.getenv('NANOLM_CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',') if x.strip()],
        allow_methods=['GET', 'POST'], allow_headers=['Content-Type'], allow_credentials=False)
    return app


app = create_app()
