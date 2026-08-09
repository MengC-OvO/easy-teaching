# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Knowledge maintenance | `ingest_knowledge.py`, `build_vector_index.py`, `query_vector_index.py` |
| Evaluation | `run_evals.py`, `run_reliability_checks.py` |

New scripts should be added only when they provide a clear
operator command; reusable implementation belongs in `app/` or `evals/`.
