PYTHON := .venv/bin/python
CUBED_CORE := .venv/bin/cubed-core
LABEL_CPU_VENV := .venv-label-cpu
LABEL_CPU_PYTHON := $(LABEL_CPU_VENV)/bin/python
LABEL_CPU_CLI := $(LABEL_CPU_VENV)/bin/cubed-core
DECODE_GPU_VENV := .venv-decode-gpu
DECODE_GPU_PYTHON := $(DECODE_GPU_VENV)/bin/python
DECODE_GPU_CLI := $(DECODE_GPU_VENV)/bin/cubed-core
TRAINING_VENV := .venv-training
TRAINING_PYTHON := $(TRAINING_VENV)/bin/python
TRAINING_BOOTSTRAP_PYTHON ?= python3
BOOTSTRAP_PYTHON ?= python3
UV ?= uv
UV_VERSION := 0.11.16
NODE ?= node
NODE_VERSION_RANGE := >=22.3.0 <23
NPM ?= npm
WEB_DIR := apps/lab-web
DOCKER := docker
COMPOSE := $(DOCKER) compose
BASE_URL ?= http://127.0.0.1:8000
RELEASE_ASSET_DIR ?= $(CURDIR)/workspace/release-assets
RELEASE_TAG ?=
RELEASE_ASSET_BASE_URL ?=
RELEASE_ASSET_SOURCE_ARGS := $(if $(RELEASE_ASSET_BASE_URL),--base-url "$(RELEASE_ASSET_BASE_URL)",$(if $(RELEASE_TAG),--tag "$(RELEASE_TAG)",))
DATASET_MANIFEST ?= $(CURDIR)/config/public-dataset-v1.json
DATASET_OUTPUT_DIR ?= $(CURDIR)/datasets/public/cubed-solves-v1
DATASET_BASE_URL ?=
DATASET_REVISION ?=
DATASET_SOURCE_ARGS := $(if $(DATASET_BASE_URL),--base-url "$(DATASET_BASE_URL)",) $(if $(DATASET_REVISION),--revision "$(DATASET_REVISION)",)
DECODE_WORKSPACE ?= $(if $(CUBED_CORE_WORKSPACE),$(CUBED_CORE_WORKSPACE),$(CURDIR)/workspace)

.DEFAULT_GOAL := help

.PHONY: help
.PHONY: api bootstrap bootstrap-compute-gpu bootstrap-decode-gpu bootstrap-label-cpu bootstrap-research-gpu bootstrap-training check
.PHONY: cloud-smoke cloud-smoke-gpu compose-check dev
.PHONY: docker-cpu docker-decode-gpu docker-gpu-check docker-label-cpu docker-workspace
.PHONY: decode-preflight doctor download-assets download-dataset download-decode-support
.PHONY: download-demo download-runtime
.PHONY: format frontend-sbom-check lint lock-check package-tracker-models provision-gpu-box
.PHONY: doctor-decode-gpu doctor-label-cpu
.PHONY: hub
.PHONY: node-version-check python-sbom-check sbom-check uv-version-check
.PHONY: release-evidence release-evidence-verify
.PHONY: remote-doctor
.PHONY: test web web-build web-test workbench
.PHONY: workbench-decode-gpu workbench-label-cpu

help:
	@echo "Cubed Core"
	@echo "  ./setup.sh              bootstrap the POSIX development environment"
	@echo "  make workbench          run the built UI and API (CUBED_CORE_PORT, default 8000)"
	@echo "  make dev                run that API and the hot-reload frontend on port 5173"
	@echo "  make hub                load ignored .env.hub settings and run the workbench"
	@echo "  make check              run the full local verification gate"
	@echo "  make docker-cpu         build and run the base Docker workbench"
	@echo "  make download-demo      fetch the demo video and its license notice"
	@echo "  make download-assets    fetch the runtime, demo, and decode-support files"
	@echo "  make download-decode-support"
	@echo "                          fetch the read-trust model, demo calibration, and license"
	@echo "  make download-dataset   fetch, verify, and register the pinned public videos for Decode"
	@echo "  make bootstrap-label-cpu"
	@echo "                          create the optional native Label model environment"
	@echo "  make bootstrap-decode-gpu"
	@echo "                          create the local CUDA Decode environment"
	@echo "  make bootstrap-research-gpu"
	@echo "                          create the full research Decode CUDA environment"
	@echo "  make bootstrap-compute-gpu"
	@echo "                          same, without Node (headless remote GPU box)"
	@echo "  make provision-gpu-box  bootstrap a remote GPU box over SSH for the"
	@echo "                          selectable Decode runner (see"
	@echo "                          scripts/provision_gpu_box.sh)"
	@echo "  make remote-doctor      non-destructive readiness check for a configured remote GPU host"
	@echo "  make decode-preflight CAPTURE=<id>"
	@echo "                          print the decode readiness checks for one local capture"
	@echo "See README.md, docs/tutorials/DECODE.md, and docs/CLOUD_GPU.md."

uv-version-check:
	@command -v "$(UV)" >/dev/null 2>&1 \
		|| (echo "uv $(UV_VERSION) is required; install it before bootstrapping" >&2; exit 2)
	@actual_version="$$("$(UV)" --version | awk '{print $$2}')"; \
	test "$$actual_version" = "$(UV_VERSION)" \
		|| (echo "uv $(UV_VERSION) is required; found $$actual_version" >&2; exit 2)

node-version-check:
	@command -v "$(NODE)" >/dev/null 2>&1 \
		|| (echo "Node $(NODE_VERSION_RANGE) is required; node was not found" >&2; exit 2)
	@actual_version="$$("$(NODE)" --version 2>/dev/null)"; \
	python3 -c 'import re, sys; actual = sys.argv[1]; match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", actual); supported = bool(match) and int(match.group(1)) == 22 and int(match.group(2)) >= 3; reported = actual or "unknown"; sys.stderr.write("" if supported else f"Node >=22.3.0 <23 is required; found {reported}\n"); raise SystemExit(0 if supported else 2)' "$$actual_version"

bootstrap: uv-version-check node-version-check
	$(BOOTSTRAP_PYTHON) -m venv .venv
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv" \
		$(UV) sync --locked --extra dev --extra label --extra decode
	npm --prefix $(WEB_DIR) ci

bootstrap-label-cpu: uv-version-check node-version-check
	$(BOOTSTRAP_PYTHON) -m venv $(LABEL_CPU_VENV)
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(LABEL_CPU_VENV)" \
		$(UV) sync --locked --extra dev --extra label --extra decode \
			--extra tracker-cpu
	$(UV) pip check --python $(LABEL_CPU_PYTHON)
	npm --prefix $(WEB_DIR) ci

bootstrap-decode-gpu: uv-version-check node-version-check
	$(BOOTSTRAP_PYTHON) -m venv $(DECODE_GPU_VENV)
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(DECODE_GPU_VENV)" \
		$(UV) sync --locked --extra dev --extra label --extra decode \
			--extra tracker-gpu --extra research-gpu
	$(UV) pip check --python $(DECODE_GPU_PYTHON)
	npm --prefix $(WEB_DIR) ci

bootstrap-research-gpu: uv-version-check node-version-check
	$(BOOTSTRAP_PYTHON) -m venv $(DECODE_GPU_VENV)
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(DECODE_GPU_VENV)" \
		$(UV) sync --locked --extra dev --extra label --extra decode \
			--extra tracker-gpu --extra research-gpu
	$(UV) pip check --python $(DECODE_GPU_PYTHON)
	npm --prefix $(WEB_DIR) ci

bootstrap-compute-gpu: uv-version-check
	$(BOOTSTRAP_PYTHON) -m venv $(DECODE_GPU_VENV)
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(DECODE_GPU_VENV)" \
		$(UV) sync --locked --extra dev --extra label --extra decode \
			--extra tracker-gpu --extra research-gpu
	$(UV) pip check --python $(DECODE_GPU_PYTHON)

bootstrap-training: uv-version-check
	$(TRAINING_BOOTSTRAP_PYTHON) -c 'import sys; supported = (3, 10) <= sys.version_info[:2] <= (3, 12); raise SystemExit(0 if supported else "training requires Python 3.10-3.12")'
	$(TRAINING_BOOTSTRAP_PYTHON) -m venv $(TRAINING_VENV)
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(TRAINING_VENV)" \
		$(UV) sync --locked --extra training
	$(UV) pip check --python $(TRAINING_PYTHON)

package-tracker-models:
	@test -n "$(SPEC)" || (echo "usage: make package-tracker-models SPEC=/private/spec.json OUTPUT=/outside/new-package" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "usage: make package-tracker-models SPEC=/private/spec.json OUTPUT=/outside/new-package" >&2; exit 2)
	$(LABEL_CPU_CLI) package-tracker-models --spec "$(SPEC)" --output-dir "$(OUTPUT)"

provision-gpu-box:
	bash scripts/provision_gpu_box.sh $(ARGS)

download-assets:
	python3 scripts/download_release_assets.py --output-dir "$(RELEASE_ASSET_DIR)" $(RELEASE_ASSET_SOURCE_ARGS)

download-runtime:
	python3 scripts/download_release_assets.py --output-dir "$(RELEASE_ASSET_DIR)" \
		--include runtime $(RELEASE_ASSET_SOURCE_ARGS)

download-demo:
	python3 scripts/download_release_assets.py --output-dir "$(RELEASE_ASSET_DIR)" \
		--include demo $(RELEASE_ASSET_SOURCE_ARGS)

download-decode-support:
	python3 scripts/download_release_assets.py --output-dir "$(RELEASE_ASSET_DIR)" \
		--include decode-support $(RELEASE_ASSET_SOURCE_ARGS)

download-dataset:
	$(PYTHON) scripts/download_dataset.py --manifest "$(DATASET_MANIFEST)" \
		--output-dir "$(DATASET_OUTPUT_DIR)" --workspace "$(DECODE_WORKSPACE)" \
		$(DATASET_SOURCE_ARGS)

doctor:
	$(CUBED_CORE) doctor

doctor-label-cpu:
	CUBED_CORE_TRACKER_MODEL_MANIFEST="$${CUBED_CORE_TRACKER_MODEL_MANIFEST:-$(CURDIR)/models/local/manifest.json}" \
	CUBED_CORE_TRACKER_ONNX_PROVIDERS=CPUExecutionProvider \
		$(LABEL_CPU_CLI) doctor

doctor-decode-gpu:
	PYTHONPATH="$(CURDIR):$(CURDIR)/scripts" \
		$(DECODE_GPU_PYTHON) -c 'import av, cv2, numpy, onnx, onnxruntime, scipy, torch'
	CUBED_CORE_TRACKER_MODEL_MANIFEST="$${CUBED_CORE_TRACKER_MODEL_MANIFEST:-$(CURDIR)/models/local/manifest.json}" \
	CUBED_CORE_TRACKER_ONNX_PROVIDERS=CUDAExecutionProvider \
		$(DECODE_GPU_CLI) doctor

remote-doctor:
	bash scripts/remote_doctor.sh

decode-preflight:
	@test -n "$(CAPTURE)" \
		|| (echo "usage: make decode-preflight CAPTURE=<capture-id> [DECODE_WORKSPACE=/path/to/workspace]" >&2; exit 2)
	$(PYTHON) scripts/decode_preflight.py --capture-id "$(CAPTURE)" --workspace "$(DECODE_WORKSPACE)" --format text

compose-check:
	$(COMPOSE) config --quiet

docker-workspace:
	@workspace_path="$${CUBED_CORE_WORKSPACE:-$(CURDIR)/workspace}"; \
	mkdir -p -- "$$workspace_path"

docker-cpu: docker-workspace
	@CUBED_CORE_UID="$${CUBED_CORE_UID:-$$(id -u)}" \
		CUBED_CORE_GID="$${CUBED_CORE_GID:-$$(id -g)}" \
		$(COMPOSE) up --build lab

docker-gpu-check:
	$(COMPOSE) --profile gpu-check run --rm gpu-check

docker-decode-gpu: docker-workspace
	@CUBED_CORE_UID="$${CUBED_CORE_UID:-$$(id -u)}" \
		CUBED_CORE_GID="$${CUBED_CORE_GID:-$$(id -g)}" \
		$(COMPOSE) --profile decode-gpu up --build lab-decode-gpu

docker-label-cpu: docker-workspace
	@CUBED_CORE_UID="$${CUBED_CORE_UID:-$$(id -u)}" \
		CUBED_CORE_GID="$${CUBED_CORE_GID:-$$(id -g)}" \
		$(COMPOSE) --profile label-cpu up --build lab-label-cpu

cloud-smoke:
	python3 scripts/cloud_smoke.py --base-url "$(BASE_URL)"

cloud-smoke-gpu:
	python3 scripts/cloud_smoke.py --base-url "$(BASE_URL)" --require-gpu

api:
	$(CUBED_CORE) serve --reload $(if $(BROWSER_PORT),--browser-port $(BROWSER_PORT),)

web: node-version-check
	npm --prefix $(WEB_DIR) run dev

dev:
	@set -eu; \
	admin_token="$${CUBED_CORE_ADMIN_TOKEN:-$$($(PYTHON) -c 'import secrets; print(secrets.token_urlsafe(32))')}"; \
	CUBED_CORE_ADMIN_TOKEN="$$admin_token" $(MAKE) api BROWSER_PORT=5173 & api_pid=$$!; \
	$(MAKE) web & web_pid=$$!; \
	trap 'kill $$api_pid $$web_pid 2>/dev/null || true' INT TERM EXIT; \
	wait $$api_pid $$web_pid

# Remote/GPU launcher: load stable settings from ignored .env.hub before
# starting the ordinary workbench.
hub:
	bash scripts/run_workbench_hub.sh

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts

web-build: node-version-check
	npm --prefix $(WEB_DIR) run build

web-test: node-version-check
	npm --prefix $(WEB_DIR) run test

workbench: web-build
	$(CUBED_CORE) serve

workbench-label-cpu: web-build
	$(LABEL_CPU_CLI) serve

workbench-decode-gpu: web-build
	$(DECODE_GPU_CLI) serve

lock-check: uv-version-check
	$(UV) lock --check

python-sbom-check: lock-check
	$(UV) export --locked --format cyclonedx1.5 --preview-features sbom-export \
		--extra dev --extra label --extra decode >/dev/null
	$(UV) export --locked --format cyclonedx1.5 --preview-features sbom-export \
		--extra dev --extra label --extra decode --extra tracker-cpu >/dev/null
	$(UV) export --locked --format cyclonedx1.5 --preview-features sbom-export \
		--extra dev --extra label --extra decode --extra tracker-gpu >/dev/null
	$(UV) export --locked --format cyclonedx1.5 --preview-features sbom-export \
		--extra dev --extra label --extra decode --extra tracker-gpu \
		--extra research-gpu >/dev/null
	$(UV) export --locked --format cyclonedx1.5 --preview-features sbom-export \
		--extra training >/dev/null

frontend-sbom-check: node-version-check
	npm --prefix $(WEB_DIR) sbom --package-lock-only \
		--sbom-format=cyclonedx >/dev/null

sbom-check: python-sbom-check frontend-sbom-check

release-evidence: uv-version-check node-version-check lock-check
	@test -n "$(SOURCE_REF)" \
		|| (echo "usage: make release-evidence SOURCE_REF=<tag-or-commit> OUTPUT=/outside/release-evidence" >&2; exit 2)
	@test -n "$(OUTPUT)" \
		|| (echo "usage: make release-evidence SOURCE_REF=<tag-or-commit> OUTPUT=/outside/release-evidence" >&2; exit 2)
	python3 scripts/build_release_evidence.py build \
		--source-ref "$(SOURCE_REF)" \
		--output-dir "$(OUTPUT)" \
		--uv "$(UV)" \
		--node "$(NODE)" \
		--npm "$(NPM)"

release-evidence-verify:
	@test -n "$(OUTPUT)" \
		|| (echo "usage: make release-evidence-verify OUTPUT=/outside/release-evidence" >&2; exit 2)
	python3 scripts/build_release_evidence.py verify --bundle-dir "$(OUTPUT)"

check: node-version-check lint test web-test web-build lock-check sbom-check

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check src tests scripts --fix
