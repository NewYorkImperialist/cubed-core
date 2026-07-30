# syntax=docker/dockerfile:1.7

# Keep multi-architecture base-image digests explicit. Override the complete
# image reference with a build arg only when deliberately updating the runtime.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d
ARG NODE_IMAGE=node:22.23.1-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

FROM ${UV_IMAGE} AS uv-bin

FROM ${NODE_IMAGE} AS lab-web
WORKDIR /build/apps/lab-web
COPY apps/lab-web/package.json apps/lab-web/package-lock.json ./
RUN npm ci
COPY apps/lab-web/ ./
RUN npm run build

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUBED_CORE_REPO_ROOT=/app \
    CUBED_CORE_WORKSPACE=/workspace \
    PATH=/app/.venv/bin:$PATH
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg zlib1g \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES ./
COPY config/ ./config/
COPY schemas/ ./schemas/
COPY detect/ ./detect/
COPY core/ ./core/
COPY analysis/ ./analysis/
COPY scripts/ ./scripts/
COPY src/ ./src/
ARG CUBED_CORE_PYTHON_EXTRAS=label,decode
RUN --mount=from=uv-bin,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    case "$CUBED_CORE_PYTHON_EXTRAS" in \
        label,decode) uv_extras="--extra label --extra decode" ;; \
        label,decode,tracker-cpu) \
            uv_extras="--extra label --extra decode --extra tracker-cpu" ;; \
        label,decode,tracker-gpu) \
            uv_extras="--extra label --extra decode --extra tracker-gpu" ;; \
        label,decode,tracker-gpu,research-gpu) \
            uv_extras="--extra label --extra decode --extra tracker-gpu --extra research-gpu" ;; \
        *) \
            echo "unsupported CUBED_CORE_PYTHON_EXTRAS=$CUBED_CORE_PYTHON_EXTRAS" >&2; \
            exit 2 ;; \
    esac; \
    uv sync --locked --no-dev --no-editable --link-mode=copy $uv_extras \
    && uv pip check --python /app/.venv/bin/python
COPY --from=lab-web /build/apps/lab-web/dist ./apps/lab-web/dist
RUN useradd --create-home --uid 10001 cubed \
    && mkdir -p /workspace \
    && chown cubed:cubed /workspace
VOLUME ["/workspace"]
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"]
USER cubed
CMD ["cubed-core", "serve", "--host", "0.0.0.0", "--port", "8000", "--allow-network"]
