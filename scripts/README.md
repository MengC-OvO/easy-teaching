# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Knowledge maintenance | `ingest_knowledge.py`, `build_vector_index.py`, `query_vector_index.py` |
| Evaluation | `run_evals.py`, `run_reliability_checks.py` |
| Live single-question check | `ask_live.py` |

Run one question through the real model, production Main ReAct graph,
PostgreSQL store, and PostgreSQL checkpoint saver:

```bash
python scripts/ask_live.py "What does the EYLF say about play-based learning?" --trace
```

This requires configured model credentials, a running PostgreSQL service, and
applied Alembic migrations. It stores the synthetic demo session and run in the
local PostgreSQL database.

New scripts should be added only when they provide a clear
operator command; reusable implementation belongs in `app/` or `evals/`.
