# Test suite

This directory contains committed regression code, not generated test output.
Keeping it in the repository makes EasyTeaching's safety and orchestration
claims reviewable and prevents later refactors from silently bypassing them.

The files cover five boundaries:

- API contracts, idempotency, SSE lifecycle, ownership, and recovery;
- Main ReAct decisions, bounded Tool/Worker execution, and GraphState behavior;
- RAG ingestion, retrieval, citations, and memory isolation;
- retry, fallback, structured-output, and failure-injection behavior;
- local privacy gateway rules, HTTP contracts, fail-closed input handling,
  redacted GraphState, deterministic restoration, and mapping cleanup.

Run everything in a fully configured development environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run only the lightweight privacy/API regression group:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_async_api.py `
  tests\test_input_safety.py `
  tests\test_privacy_gateway_api.py `
  tests\test_privacy_gateway_client.py `
  tests\test_safety_gateway_pipeline.py
```

Pytest caches, coverage output, logs, generated reports, model assets, local
databases, and virtual environments are ignored. Test source files are not.
