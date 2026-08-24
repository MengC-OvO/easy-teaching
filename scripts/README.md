# Script guide

Scripts are local entry points, not production API modules.

| Group | Scripts |
| --- | --- |
| Knowledge maintenance | `ingest_knowledge.py`, `build_lexical_index.py`, `build_vector_index.py`, `query_vector_index.py`, `test_rag_retrieval.py` |
| Evaluation | `run_evals.py`, `run_reliability_checks.py`, `run_rag_evals.py` |
| Live single-question check | `ask_live.py` |
| Local safety setup/runtime | `setup_safety_gateway.ps1` / `.sh`, `start_safety_gateway.ps1` / `.sh` |
| Synthetic privacy demonstration | `demo_privacy_flow.ps1` / `.sh` |

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

On an Apple Silicon Mac, run the equivalent demonstration with:

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
