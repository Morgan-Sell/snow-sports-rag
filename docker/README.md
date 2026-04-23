# Docker

## Build

First build downloads PyTorch and other deps (CPU wheels when `UV_TORCH_BACKEND=cpu` is set in the `Dockerfile`); expect several minutes and several GB.

```bash
docker build -t snow-sports-rag .
```

## Run (Compose)

From the repository root (optionally create a `.env` with `OPENAI_API_KEY` and other secrets referenced by `configs/default.yaml`):

```bash
docker compose up --build
```

Open `http://localhost:7860`. The UI uses `GRADIO_SERVER_NAME=0.0.0.0` so the server listens on all interfaces inside the container.

**Volumes**

- `knowledge-base` — read-only mount of your local corpus (override the image copy).
- Named volumes for `rag_index` (Chroma under `.rag_index`), `rag_traces` (JSONL traces), and `hf_cache` (`HF_HOME=/data/hf`) so embeddings and models survive restarts.

**Index** — Build the vector index before first query (e.g. run a one-off index command), or use `snow-sports-rag-ui --auto-index` in an overridden command if you want startup indexing (slow on first boot).

**One-off commands** (same image):

```bash
docker compose run --rm ui snow-sports-rag index --base-dir /app
docker compose run --rm ui snow-sports-rag-sweep --base-dir /app
docker compose run --rm ui snow-sports-rag-trace-analyze --traces /app/.rag_traces/traces.jsonl
```

## Environment

- Do not bake API keys into the image; pass `.env` or host environment.
- `SNOW_SPORTS_RAG_KNOWLEDGE_BASE_PATH` can point at a mounted corpus if you change layout.
