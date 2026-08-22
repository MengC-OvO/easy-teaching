#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup script requires an Apple Silicon Mac." >&2
  exit 1
fi
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/Qwen2.5-1.5B-Instruct /path/to/best-adapter" >&2
  exit 1
fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
model_dir="$(cd "$1" && pwd)"
adapter_dir="$(cd "$2" && pwd)"

[[ -f "$model_dir/model.safetensors" ]] || { echo "Missing model.safetensors" >&2; exit 1; }
[[ -f "$adapter_dir/adapter_model.safetensors" ]] || { echo "Missing adapter_model.safetensors" >&2; exit 1; }

if [[ ! -x "$project_root/.venv-safety/bin/python" ]]; then
  echo "Creating .venv-safety (expect roughly 1-2 GB, excluding model files)..."
  python3 -m venv "$project_root/.venv-safety"
fi

python_bin="$project_root/.venv-safety/bin/python"
"$python_bin" -m pip install \
  -r "$project_root/safety_gateway/requirements.txt" \
  -r "$project_root/safety_gateway/requirements-model-mac.txt"

cat > "$project_root/.safety-gateway.env" <<EOF
SAFETY_MODEL_BACKEND=auto
SAFETY_MODEL_DIR="$model_dir"
SAFETY_ADAPTER_DIR="$adapter_dir"
SAFETY_MAX_INPUT_TOKENS=1536
SAFETY_MAX_NEW_TOKENS=320
SAFETY_MAPPING_TTL_SECONDS=3600
EOF

echo "Safety Gateway environment is ready."
echo "Start with: ./scripts/start_safety_gateway.sh"
