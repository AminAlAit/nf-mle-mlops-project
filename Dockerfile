# Stage 1: borrow the uv binary from its official image
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv

# Stage 2: the actual API image
FROM python:3.13-slim-bookworm

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/

# Install dependencies first (cached layer: only rebuilt when the lock changes)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Then copy your code
COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]