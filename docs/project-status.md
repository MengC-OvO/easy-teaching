# Project status

Updated: 2026-08-26

## Implemented

- FastAPI web application with a browser chat workspace, durable sessions,
  idempotent message requests and replayable Agent-run events.
- PostgreSQL-only async business store and LangGraph checkpoints, managed by
  Docker Compose and Alembic migrations.
- One bounded Main ReAct loop with validated single Tools, concurrent independent
  Tools, and exactly two read-only Worker profiles for paired deep research.
- Class context, record search, draft-artifact loading, activity safety,
  weather/holiday context, hybrid RAG, observation/educational-record save,
  export, and optional Google Drive search/upload Tools.
- Frozen-argument, teacher-approved controlled writes with atomic exactly-once
  execution; no real family-message sending capability.
- Layout-aware knowledge ingestion, 2,087 chunks, SQLite FTS5/BM25, Chroma dense
  search, hard source filtering, weighted RRF, optional local Cross-encoder, and
  citation metadata.
- Compact short-term context, scoped long-term teacher/class memory, full draft
  artifact references, retry/fallback behavior and durable failure recovery.
- Automated regression, RAG metrics, final Agent evaluation and an independent
  local privacy/safety Gateway evaluation.

## Evidence

- Repository regression at final evaluation: `329/329` passed.
- Final Agent: `39/40` scenarios (`97.5%`), corrected isolated rerun `1/1`.
- Required-Tool recall `100%`, Tool precision `95.8%`, forbidden calls `0%`,
  approval integrity `100%`, path efficiency `91.1%`.
- RAG Hybrid + Cross-encoder: Recall@3 `0.975`, MRR `0.869`, nDCG@10 `0.902`,
  scope violations `0`, citation correctness `1.0` on the frozen 40-case set.
- Local privacy Gateway release gate: **failed**; PII recall `60%`, PII F1
  `75%`, phone recall `0%`, with fail-closed behavior and no plaintext response
  leakage.

## Deliberately out of scope

- Real child/family data while the privacy Gateway release gate is failing.
- Production multi-tenancy, centre-level RBAC, encryption/key management,
  retention/deletion governance and independent security review.
- Redis-backed shared ephemeral state, distributed rate limiting, background job
  workers, horizontal API scaling, production monitoring and load certification.
- Real outbound family messaging and autonomous unapproved writes.

## Recommended next production phase

1. Retrain the local adapter using the exact premasked serving distribution and
   pass the frozen Gateway suite plus independent human/adversarial review.
2. Implement authenticated centre tenancy and role-based authorization at every
   database and Tool boundary.
3. Add Redis for shared cache/ephemeral coordination and a queue worker for
   ingestion, exports, Drive upload and other long-running jobs.
4. Add secrets management, encrypted sensitive storage, audit retention,
   backups, observability, cost limits and concurrent load tests.

This file is the release snapshot. Architecture rationale and operational detail
remain in `architecture.md`, `api-and-operations.md`, `tool-architecture.md`,
`rag-system.md`, `agent-evaluation.md`, and `local-safety-gateway.md`.
