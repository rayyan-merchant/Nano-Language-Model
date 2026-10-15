FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NANOLM_ARTIFACTS=/models NANOLM_DEVICE=cpu NANOLM_CPU_THREADS=1
WORKDIR /app
COPY requirements.txt requirements-studio.txt pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt -r requirements-studio.txt
COPY src ./src
RUN python -m pip install --no-cache-dir -e . --no-deps
COPY api ./api
COPY configs ./configs
COPY experiments/configs ./experiments/configs
COPY assets/tokenizers ./assets/tokenizers
COPY report/pilots/E02_seed42/evidence/artifact_manifest.json ./report/pilots/E02_seed42/evidence/artifact_manifest.json
COPY report/balanced/evidence/research/artifact_manifest.json ./report/balanced/evidence/research/artifact_manifest.json
COPY report/final/evidence/research/artifact_manifest.json ./report/final/evidence/research/artifact_manifest.json
COPY report/pilots/E02_seed42/summary.json ./report/pilots/E02_seed42/summary.json
COPY report/balanced/summary.json ./report/balanced/summary.json
COPY report/final/summary.json ./report/final/summary.json
RUN useradd --create-home --uid 10001 studio && chown -R studio:studio /app
USER studio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--no-proxy-headers"]
