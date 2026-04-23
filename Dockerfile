# RAG app + Gradio UI. Build: docker build -t snow-sports-rag .
# Run: see docker/README.md
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Avoid CUDA PyTorch wheels in Linux images (saves multi-GB downloads).
ENV UV_TORCH_BACKEND=cpu

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/data/hf

EXPOSE 7860

CMD ["snow-sports-rag-ui", "--host", "0.0.0.0", "--port", "7860"]
