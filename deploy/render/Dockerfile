# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    MANTIS_HOST=0.0.0.0 \
    MANTIS_CONDUCTOR_DEVICE=cpu \
    MANTIS_CONDUCTOR_DTYPE=float32 \
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

# Build venvs with uv
RUN uv sync --locked --no-dev
RUN cd apps/gateway && uv sync --no-dev

# Copy the rest of the application
COPY . .

# Ensure vector artifacts exist
RUN if [ ! -f "artifacts/model_iter_60.npy" ]; then \
      uv run --no-sync python scripts/make_vec.py --mode head-only --input artifacts/router_head.safetensors --output artifacts/model_iter_60.npy; \
    fi

RUN chmod +x deploy/render/render-entrypoint.sh scripts/run_mantis_native.sh

EXPOSE 8088 10000

ENTRYPOINT ["/app/deploy/render/render-entrypoint.sh"]
