# Agent evaluation

Updated: 2026-08-26

## What is evaluated

EasyTeaching keeps component tests separate from the final Agent benchmark so
one aggregate score does not hide the source of a failure.

| Layer | Command | Purpose |
| --- | --- | --- |
| Unit and contract regression | `python -m pytest` | API, graph, Tool, approval, RAG, memory and privacy contracts |
| Offline Agent suite | `python scripts/run_evals.py` | deterministic routing and failure behavior without provider cost |
| RAG retrieval | `python scripts/run_rag_evals.py` | Recall@K, MRR, nDCG, scope, citations and retrieval latency |
| Final live Agent suite | `python scripts/run_final_agent_suite.py` | real HTTP, PostgreSQL, LangGraph, Gemini, Tools/Workers and local RAG |
| Local privacy model | `python scripts/run_local_safety_final_eval.py` | independent Gateway, mapping, restoration, PII and injection evaluation |

The final Agent suite contains 40 synthetic scenarios and 46 conversational
turns across activity safety, EYLF/NQS/policy RAG, records, frozen controlled
writes, family communication, orchestration, security, multi-turn draft
selection, idempotency and concurrency. Weather and Google Drive use
deterministic adapters so external uptime and personal files cannot change the
benchmark. Authentication and the local privacy model are evaluated separately.

## Final result — 2026-08-25

| Metric | Result |
| --- | ---: |
| Full-run scenario pass | `39/40 (97.5%)` |
| Required-Tool recall | `100.0%` |
| Tool precision (micro) | `95.8%` |
| Derived Tool-selection F1 | `97.9%` |
| Tool parameter-contract accuracy | `100.0%` |
| Forbidden-Tool violation rate | `0.0%` |
| Average path efficiency | `91.1%` |
| Approval integrity | `100.0%` |
| RAG grounding | `100.0%` |
| Security / multi-turn / operational checks | `100% / 100% / 100%` |
| LLM-judged answer quality | `19/19`, mean `98.9%` |
| ReAct steps P50 / P95 / maximum | `2 / 4 / 4` |
| Step-limit exhaustion / fallback | `0% / 0%` |
| Decision-feedback rate | `2.4%` |
| Latency P50 / P95 | `3.46 s / 11.68 s` |
| Model calls | `118` |
| Prompt / completion / total tokens | `561,496 / 22,805 / 584,301` |
| Offline regression at evaluation time | `329/329` passed |

The only full-run failure was a bad assertion: one Chinese record-query case
required the answer to expose an internal random marker. The Agent correctly
retrieved the marker-scoped record and summarized its facts. The assertion was
changed to test those facts, and the isolated rerun passed `1/1`. The original
report is retained rather than rewritten.

## What the numbers mean

- `100%` Tool recall means every required capability was selected.
- `95.8%` Tool precision means a small number of valid but unnecessary calls
  remained; this is an efficiency issue, not a permission violation.
- `91.1%` path efficiency and a four-step maximum show the bounded ReAct loop
  converges reliably on this dataset.
- `0%` forbidden calls and `100%` approval integrity show deterministic code
  boundaries held even when the model chose the trajectory.
- P95 latency and prompt tokens are the main optimization targets. Roughly 96%
  of measured tokens were prompt input, reflecting repeated context and Tool
  schemas rather than unusually long final answers.

This result validates the current production-path implementation against the
versioned synthetic suite. It is not a production safety certification: the set
is relatively small and partly judged by the same provider family. Deployment
acceptance requires independently authored cases, human review, load tests, real
tenant isolation, cost monitoring, and a larger adversarial set.

## Reproduce

```powershell
# No provider cost
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_evals.py

# Uses configured Gemini and embedding quota
.\.venv\Scripts\python.exe scripts\run_final_agent_suite.py `
  --concurrency 3 `
  --output reports\final_agent_evaluation.json

# Debug one scenario
.\.venv\Scripts\python.exe scripts\run_final_agent_suite.py `
  --case SCENARIO_ID `
  --output reports\final_agent_focus.json
```

Generated reports are intentionally ignored by Git. The final local evidence is
summarized in `reports/agent_final_metrics_20260825.md`; it is not a committed
project dependency.
