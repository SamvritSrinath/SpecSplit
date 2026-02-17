# =============================================================================
# SpecSplit — Disaggregated Speculative Decoding
# Multi-stage Dockerfile for PyTorch inference workers
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies and compile proto stubs
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev \
        build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip — Ubuntu 22.04 ships pip 22.0 which can't resolve
# PEP 621 pyproject.toml dependencies reliably
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /build

# ---- Copy ALL project files needed for installation ----
COPY pyproject.toml README.md Makefile ./
COPY specsplit/ specsplit/
COPY tests/ tests/
COPY scripts/ scripts/

# ---- Install the package + all dependencies ----
# Using --verbose so we can see what pip is actually doing
RUN python -m pip install --no-cache-dir ".[dev]" \
    && python -c "import torch; import pydantic; import grpc_tools; print('All core deps OK')"

# ---- Compile protobuf stubs ----
RUN python -m grpc_tools.protoc \
        --proto_path=specsplit/proto \
        --python_out=specsplit/proto \
        --grpc_python_out=specsplit/proto \
        --mypy_out=specsplit/proto \
        specsplit/proto/spec_decoding.proto

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to serve
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and source from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

WORKDIR /app

# Expose gRPC ports for draft (50051) and target (50052) workers
EXPOSE 50051 50052

# Default entrypoint — overridden per-service in docker-compose.yml
ENTRYPOINT ["python"]
