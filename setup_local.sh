#!/usr/bin/env bash
set -euo pipefail

# Smart-MVP local dev bootstrapper.
#
# Goals:
# - Fast "inner loop" on Apple Silicon: run natively (no Docker) during feature dev.
# - Decouple model weights from code: keep weights in a persistent cache folder and
#   symlink into the repo as ./models (and optionally mount to /models in containers).
# - Use `uv` everywhere (project standard).
#
# Usage:
#   ./setup_local.sh              # sets up both agent + analysis-worker envs
#   ./setup_local.sh agent        # only LiveKit agent env
#   ./setup_local.sh worker       # only analysis-worker env
#
# Environment knobs:
#   MCH_MODELS_HOME=/path/to/persistent/models   # overrides default models cache location

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

say() { printf "\n==> %s\n" "$*"; }
die() { printf "\nERROR: %s\n" "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null 2>&1 || die "uv installation failed; open a new shell and retry."
}

default_models_home() {
  # Keep models out of the repo to avoid accidental commits and to preserve cache between rebuilds.
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    echo "${HOME}/Library/Caches/mch/models"
  else
    echo "${HOME}/.cache/mch/models"
  fi
}

setup_models_symlink() {
  local models_home="${MCH_MODELS_HOME:-$(default_models_home)}"
  local link_path="${ROOT_DIR}/models"

  say "Setting up persistent models cache: ${models_home}"
  mkdir -p "${models_home}"

  if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
    die "${link_path} exists and is not a symlink. Move it aside and re-run (we want ./models -> persistent cache)."
  fi

  if [[ -L "${link_path}" ]]; then
    # Keep it if it already points somewhere; just print for clarity.
    say "models symlink already exists: $(readlink "${link_path}")"
  else
    ln -s "${models_home}" "${link_path}"
    say "Created symlink: ${link_path} -> ${models_home}"
  fi

  # Optional: encourage /models usage without requiring sudo on macOS.
  # Code will prefer /models if it exists (e.g., in Docker), otherwise fall back to ./models.
}

setup_agent_env() {
  say "Bootstrapping LiveKit agent env (root .venv) with uv..."
  cd "${ROOT_DIR}"
  uv sync
  say "Agent env ready. Run: uv run python src/agent.py dev"
}

setup_worker_env() {
  # analysis-worker is intentionally kept as a separate env because its deps are heavy (torch, mediapipe, etc.)
  say "Bootstrapping analysis-worker env (analysis-worker/.venv) with uv..."
  cd "${ROOT_DIR}/analysis-worker"

  uv venv .venv

  # Install PyTorch for the current architecture.
  # - On Apple Silicon (darwin/arm64), pip wheels include MPS support; no CPU index needed.
  # - On linux/amd64, we default to CPU wheels to keep the baseline simple and serverless-friendly.
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    say "Installing PyTorch (Apple Silicon / MPS-capable wheels)..."
    uv pip install torch torchvision torchaudio
  else
    say "Installing PyTorch (linux CPU wheels)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi

  say "Installing analysis-worker Python deps..."
  uv pip install -r requirements.txt

  say "analysis-worker env ready. Run: ./scripts/run_sandbox.sh (after we add it) or: python analyze_sandbox.py ..."
}

ensure_uv
setup_models_symlink

case "${TARGET}" in
  all)
    setup_agent_env
    setup_worker_env
    ;;
  agent)
    setup_agent_env
    ;;
  worker)
    setup_worker_env
    ;;
  *)
    die "Unknown target '${TARGET}'. Use: all | agent | worker"
    ;;
esac

say "Done."


