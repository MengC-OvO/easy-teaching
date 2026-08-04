# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Application demos | `live_api_llm_demo.py`, `week1_demo.py`, `gemini_week1_trace.py` |
| Model and graph smoke tests | `model_smoke_test.py`, `intent_router_smoke_test.py`, `react_smoke_test.py`, `policy_rag_smoke_test.py`, `real_policy_rag_answer.py` |
| Knowledge maintenance | `ingest_knowledge.py`, `build_vector_index.py`, `query_vector_index.py` |
| Database migration | `migrate_sqlite_to_postgres.py` |
| Evaluation | `run_week2_evals.py`, `run_evals.py`, `run_reliability_checks.py` |

They remain at the top of `scripts/` so existing commands, tests, and learning
notes keep working. New scripts should be added only when they provide a clear
operator command; reusable implementation belongs in `app/` or `evals/`.
