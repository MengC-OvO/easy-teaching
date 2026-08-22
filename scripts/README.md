# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Knowledge maintenance | `ingest_knowledge.py`, `build_vector_index.py`, `query_vector_index.py` |
| Evaluation | `run_evals.py`, `run_reliability_checks.py` |
| Live single-question check | `ask_live.py` |
| Local safety setup/runtime | `setup_safety_gateway.ps1`, `start_safety_gateway.ps1` |
| Synthetic privacy demonstration | `demo_privacy_flow.ps1` |

Run one question through the real model, production Main ReAct graph,
PostgreSQL store, and PostgreSQL checkpoint saver:

```bash
python scripts/ask_live.py "What does the EYLF say about play-based learning?" --trace
```

This requires configured model credentials, a running PostgreSQL service, and
applied Alembic migrations. It stores the synthetic demo session and run in the
local PostgreSQL database.

Run the full local privacy demonstration from PowerShell:

```powershell
.\scripts\demo_privacy_flow.ps1
```

This command uses the real local Qwen adapter and production privacy/FastAPI
boundaries with fixed synthetic input. `_privacy_flow_demo.py` is its internal
Python implementation, not a separate operator command. The ReAct node is
deterministic so the demonstration does not require PostgreSQL or an external
model-provider credential.

New scripts should be added only when they provide a clear
operator command; reusable implementation belongs in `app/` or `evals/`.
