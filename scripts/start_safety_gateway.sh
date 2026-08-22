#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$project_root/.venv-safety/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing .venv-safety; run setup_safety_gateway.sh first." >&2; exit 1; }

cd "$project_root"
echo "Starting EasyTeaching Local Safety Gateway on 127.0.0.1:8010"
exec "$python_bin" -m uvicorn safety_gateway.runtime:app --host 127.0.0.1 --port 8010 --workers 1
