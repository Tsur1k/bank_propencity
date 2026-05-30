FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache-dir -r pyproject.toml

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

RUN useradd -m mluser && chown -R mluser:mluser /app
USER mluser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import lightgbm; import mlflow; print('OK')" || exit 1

CMD ["python", "-m", "src.train", "--config", "configs/pipeline.yaml"]