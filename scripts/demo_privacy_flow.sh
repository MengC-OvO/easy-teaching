#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
safety_python="$project_root/.venv-safety/bin/python"
app_python="$project_root/.venv/bin/python"
gateway_url="http://127.0.0.1:8010"
gateway_pid=""

[[ -x "$safety_python" ]] || { echo "Missing .venv-safety; run setup_safety_gateway.sh first." >&2; exit 1; }
[[ -x "$app_python" ]] || { echo "Missing main .venv." >&2; exit 1; }
[[ -f "$project_root/.safety-gateway.env" ]] || { echo "Missing .safety-gateway.env." >&2; exit 1; }

cleanup() {
  if [[ -n "$gateway_pid" ]]; then
    kill "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready() {
  curl --silent --fail "$gateway_url/ready" >/dev/null 2>&1
}

echo "============================================================"
echo " EasyTeaching full local privacy-flow smoke test"
echo " Real Qwen adapter: YES | External LLM: NO | Real data: NO"
echo "============================================================"

if ready; then
  echo "[START] Safety Gateway is already running; it will be reused."
else
  echo "[START] Launching the local Safety Gateway in the background..."
  (
    cd "$project_root"
    "$safety_python" -m uvicorn safety_gateway.runtime:app \
      --host 127.0.0.1 --port 8010 --workers 1
  ) &
  gateway_pid=$!

  for attempt in {1..36}; do
    if ready; then break; fi
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      echo "Safety Gateway exited while loading the model." >&2
      exit 1
    fi
    echo "[LOAD] Qwen base model + LoRA are loading... check $attempt"
    sleep 5
  done
  ready || { echo "Timed out waiting for the gateway." >&2; exit 1; }
fi

echo "[READY] Local Qwen and the LoRA adapter are ready."
cd "$project_root"
"$app_python" -m scripts._privacy_flow_demo --gateway-url "$gateway_url"
