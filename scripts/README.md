# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Knowledge maintenance | `ingest_knowledge.py`, `build_lexical_index.py`, `build_vector_index.py`, `query_vector_index.py`, `test_rag_retrieval.py` |
| Evaluation | `run_evals.py`, `run_reliability_checks.py`, `run_rag_evals.py` |
| Live end-to-end Agent evaluation | `run_agent_evals.py` |
| Independent final Agent evaluation | `run_final_agent_suite.py` |
| Production-path online evaluation | `run_production_online_eval.py` |
| Live single-question check | `ask_live.py` |
| Phase-one Agent/tool smoke suite | `test_phase1_live.py` |
| Google Drive MCP connectivity | `test_google_drive_mcp.py` |
| Local safety setup/runtime | `setup_safety_gateway.ps1` / `.sh`, `start_safety_gateway.ps1` / `.sh` |
| Synthetic privacy integration verification | `demo_privacy_flow.ps1` / `.sh` |
| Final local safety evaluation | `run_local_safety_final_eval.py`, `diagnose_safety_premask_contract.py` |

The phase-one smoke suite includes `draft_save_followup`, a two-turn live check
that generates an activity and then asks the same Main ReAct thread to save the
previous draft through a frozen `save_educational_record` approval.

It also includes `versioned_draft_save`, a three-turn check that creates two
different drafts and verifies that asking to save the first version freezes the
first draft rather than the latest one. The script never auto-approves the write.

`test_artifact_selection_live.py` is the privacy-safe synthetic counterpart. It
uses the real chat model to check first/previous/latest/ambiguous reference
selection without reading PostgreSQL conversation content. Unit tests separately
verify that the chosen immutable ID resolves and freezes the exact full draft.

The production-path runner uses real FastAPI, PostgreSQL, checkpoint, RAG and
model boundaries with synthetic data. Use repeated `--case` flags for focused
rechecks; available groups are printed by `--help`. A full run intentionally
executes one approved synthetic observation write so persistence is not mocked.

For the full Agent evaluation profile, add `--include-long-term-memory` and
`--quality-judge`. The JSON report records that weather and Google Drive are
deterministic adapters, so the result does not overstate external-service coverage.

Run one question through the real model, production Main ReAct graph,
PostgreSQL store, and PostgreSQL checkpoint saver:

```bash
python scripts/ask_live.py "What does the EYLF say about play-based learning?" --trace
```

This requires configured model credentials, a running PostgreSQL service, and
applied Alembic migrations. It stores the synthetic verification session and run in the
local PostgreSQL database.

Run the full local privacy integration verification from PowerShell:

```powershell
.\scripts\demo_privacy_flow.ps1
```

This command uses the real local Qwen adapter and production privacy/FastAPI
boundaries with fixed synthetic input. `_privacy_flow_demo.py` is its internal
Python implementation, not a separate operator command. The ReAct node is
deterministic so this verification does not require PostgreSQL or an external
model-provider credential.

On an Apple Silicon Mac, run the equivalent verification with:

```bash
bash ./scripts/demo_privacy_flow.sh
```

New scripts should be added only when they provide a clear
operator command; reusable implementation belongs in `app/` or `evals/`.

Run one retrieval-only RAG check without starting the API or full Agent:

```powershell
.\.venv\Scripts\python.exe scripts\test_rag_retrieval.py `
  "What does the EYLF say about play-based learning?" `
  --scope eylf --mode hybrid
```

Use `--mode bm25` for a fully local, zero-API check. Dense and hybrid modes use
one query-embedding request. The command fails when a scoped query returns a
source outside its requested boundary.

Run the labelled retrieval metric suite without starting the full Agent:

```powershell
# Free local baseline
.\.venv\Scripts\python.exe scripts\run_rag_evals.py --modes bm25

# Compare all retrieval modes; query vectors are cached between modes and runs
.\.venv\Scripts\python.exe scripts\run_rag_evals.py --modes bm25 dense hybrid
```

The suite reports Recall@1/3/5/10, MRR, nDCG@10, source-scope violations,
citation correctness, and retrieval-only P50/P95 latency. Query embeddings are
prepared before timed retrieval so Dense and Hybrid latency remain comparable;
new API embedding latency is reported separately. Detailed results are written
to `reports/rag_retrieval_report.json`; cached query vectors stay under ignored
`data/local/`.

Run the frozen 40-case final suite (18 EYLF, 18 NQS, 4 synthetic centre-policy):

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evals.py `
  --cases data/evals/rag_final_cases.json `
  --split test --modes bm25 dense hybrid `
  --report reports/rag_final_report.json
```

The final suite uses exact `chunk_id` ground truth because the production
chunking configuration is frozen. The smaller `rag_retrieval_cases.json` file
remains the development smoke set and should not replace the final report.

Run the independent final Agent suite through the real production path:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_agent_suite.py `
  --concurrency 3 `
  --output reports\final_agent_evaluation.json
```

This consumes configured model and embedding quota. It covers 40 scenarios and
46 turns; generated reports remain ignored. The local safety evaluation must be
run only while the separate Qwen Gateway is ready on port 8010. Its current
release gate fails, so it is an evaluation command rather than a production
enablement step.
