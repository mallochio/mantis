# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    MANTIS_HOST=0.0.0.0 \
    MANTIS_CONDUCTOR_DEVICE=cpu \
    MANTIS_CONDUCTOR_DTYPE=float32 \
    MANTIS_ROUTER_DTYPE=bfloat16 \
    MANTIS_MODEL=/opt/models/qwen3-0.6b \
    AI_ROUTING_CONFIG=/app/config/catalog.toml \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iptables \
    iproute2 \
    nodejs \
    npm \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Install Bifrost globally via npm
RUN npm install -g @maximhq/bifrost

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml uv.lock ./
COPY apps/gateway/pyproject.toml apps/gateway/uv.lock apps/gateway/

# Build venvs with uv into /opt (outside the copied tree so the later
# COPY . . can never clobber them with a host venv).
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/mantis-venv uv sync --locked --no-dev
RUN --mount=type=cache,target=/root/.cache/uv \
    cd apps/gateway && UV_PROJECT_ENVIRONMENT=/opt/gateway-venv uv sync --no-dev

# Trinity uses Qwen3-0.6B as its coordinator/router backbone. Bake it into the
# image so the first Trinity request does not depend on a slow runtime download
# into Render's ephemeral filesystem.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    /opt/mantis-venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='/opt/models/qwen3-0.6b')"

# Copy the rest of the application
COPY . .

# Defensively remove any host-compatible venvs that might have slipped in
# via the build context before this layer.
RUN rm -rf /app/.venv /app/apps/gateway/.venv

# Ensure vector artifacts exist
RUN if [ ! -f "artifacts/model_iter_60.npy" ]; then \
      /opt/mantis-venv/bin/python scripts/make_vec.py --mode head-only --input artifacts/router_head.safetensors --output artifacts/model_iter_60.npy; \
    fi

RUN chmod +x deploy/render/render-entrypoint.sh scripts/run_mantis_native.sh

EXPOSE 8088 10000

ENTRYPOINT ["/app/deploy/render/render-entrypoint.sh"]
