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
| Direct local privacy model | `python scripts/run_local_model_final_eval.py` | raw-input Qwen PII and injection evaluation without regex assistance |
| Local privacy Gateway | `python scripts/run_local_safety_final_eval.py` | deployed premasking, HTTP, mapping and restoration evaluation |

The final Agent suite contains 100 synthetic scenarios and 118 conversational
turns across activity safety, EYLF/NQS/policy RAG, records, frozen controlled
writes, family communication, orchestration, security, multi-turn draft
selection, idempotency and concurrency. Weather and Google Drive use
deterministic adapters so external uptime and personal files cannot change the
benchmark. Authentication and the local privacy model are evaluated separately.

## Final result — 2026-08-26

| Metric | Result |
| --- | ---: |
| Full-run scenario pass | `93/100 (93.0%)` |
| Turn pass | `110/118 (93.2%)` |
| Required-Tool recall | `98.0%` |
| Tool precision (micro) | `92.8%` |
| Tool parameter-contract accuracy | `100.0%` |
| Forbidden-Tool violation rate | `0.8%` |
| Average path efficiency | `89.6%` |
| Approval integrity | `87.5%` |
| RAG grounding | `100.0%` |
| Security / multi-turn / operational checks | `100% / 100% / 100%` |
| LLM-judged answer quality | `57/58 (98.3%)`, mean `98.6%` |
| ReAct steps P50 / P95 / maximum | `2 / 4 / 4` |
| Step-limit exhaustion / fallback | `0% / 0%` |
| Decision-feedback rate | `0.9%` |
| Latency P50 / P95 | `3.16 s / 8.66 s` |
| Model calls | `298` |
| Prompt / completion / total tokens | `1,404,605 / 57,128 / 1,461,733` |
| Offline regression at evaluation time | `329/329` passed |

The raw result is retained without post-hoc score changes. Review of the seven
failed scenarios found three evaluator false negatives (one narrow lexical
assertion and two citation-attribution parser misses), three Agent behavior
defects (save intent lost after prerequisite reads, incomplete export-to-Drive
chaining, and missing send-capability disclosure), and one mixed isolation and
grounding defect in the recent-record summary. One policy-aligned activity also
revealed that a safety `needs_revision` observation was not fully reflected in
the final answer even though the scenario's recorded failure was citation
attribution. These findings remain tracked as defects rather than being hidden
by weakening the suite.

## What the numbers mean

- `98.0%` Tool recall shows capability selection is strong, but the missed save
  and export actions are important because they occur on controlled writes.
- `92.8%` Tool precision includes unnecessary record/Drive lookups in one
  multi-stage request; most other trajectories selected only relevant Tools.
- `89.6%` path efficiency and a four-step maximum show the bounded ReAct loop
  converges reliably on this dataset.
- `100%` parameter-contract accuracy, zero step-limit exhaustion and zero model
  fallback show the execution contracts remained stable.
- `100%` RAG, security, multi-turn and operational rates are the strongest
  results; controlled writes and communication remain the priority fixes.
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
  --output reports\final_agent_evaluation_v2.json

# Debug one scenario
.\.venv\Scripts\python.exe scripts\run_final_agent_suite.py `
  --case SCENARIO_ID `
  --output reports\final_agent_focus.json
```

Generated reports are intentionally ignored by Git. The final local evidence is
summarized in `reports/agent_final_metrics_20260825.md`; it is not a committed
project dependency.
