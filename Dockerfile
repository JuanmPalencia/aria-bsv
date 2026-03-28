# ============================================================================
# ARIA-BSV — Production Dockerfile
# ============================================================================
# Multi-stage build:
#   builder  — install deps + compile Python bytecode
#   runtime  — minimal image with only what's needed to run
#
# Build:
#   docker build -t aria-bsv:latest .
#
# Run portal API:
#   docker run -p 8080:8080 -e BSV_WIF=<key> -v $(pwd)/data:/data aria-bsv:latest
#
# Run watchdog only:
#   docker run -e BSV_WIF=<key> aria-bsv:latest python -m aria.watchdog
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps needed at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (better layer caching)
COPY pyproject.toml ./
COPY aria/__init__.py aria/

# Install project + core extras into /install prefix
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir ".[postgres,prometheus]" \
    || pip install --prefix=/install --no-cache-dir . \
    && echo "Dependencies installed"

# Copy full source
COPY . .

# Pre-compile Python bytecode
RUN python -m compileall -q aria/ portal/ registry/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="ARIA-BSV"
LABEL org.opencontainers.image.description="Auditable AI accountability layer for BSV"
LABEL org.opencontainers.image.source="https://github.com/juanm/aria-bsv"

# Non-root user for security
RUN groupadd -r aria && useradd -r -g aria -d /app -s /sbin/nologin aria

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --from=builder /build/aria        ./aria
COPY --from=builder /build/portal      ./portal
COPY --from=builder /build/registry    ./registry
COPY --from=builder /build/migrations  ./migrations
COPY --from=builder /build/alembic.ini ./alembic.ini

# Data directory (mount a volume here in production)
RUN mkdir -p /data && chown aria:aria /data

# Switch to non-root
USER aria

# Environment defaults
ENV ARIA_DB_PATH=/data/aria.db \
    ARIA_SYSTEM_ID=aria-default \
    ARIA_NETWORK=mainnet \
    ARIA_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import aria; print('ok')" || exit 1

# Default: start the portal API
CMD ["python", "-m", "uvicorn", "portal.backend.api:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--log-level", "info"]
