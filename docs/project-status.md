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

- Repository regression at current evaluation: `331/331` passed.
- Final Agent: `93/100` scenarios (`93.0%`) across `118` turns; release gate
  passed, with the raw report retained for failure analysis.
- Required-Tool recall `98.0%`, Tool precision `92.8%`, parameter-contract
  accuracy `100%`, path efficiency `89.6%`.
- RAG, security, multi-turn and operational category checks each passed `100%`;
  controlled-write chaining and capability disclosure remain tracked defects.
- RAG Hybrid + Cross-encoder: Recall@3 `0.975`, MRR `0.869`, nDCG@10 `0.902`,
  scope violations `0`, citation correctness `1.0` on the frozen 40-case set.
- Direct local-model raw-input suite: `1,227` cases, injection Macro-F1 `95.2%`,
  PII precision/recall/F1 `98.5% / 93.9% / 96.1%`; release gate **failed** on
  strict output (`98.9%`), PII entity leakage (`5.5%`) and block escape (`1.37%`).
- The separate deployed-Gateway suite also remains failed; malformed model output
  fails closed and no plaintext is returned in Gateway error responses.

## Deliberately out of scope

- Real child/family data while the privacy Gateway release gate is failing.
- Production multi-tenancy, centre-level RBAC, encryption/key management,
  retention/deletion governance and independent security review.
- Redis-backed shared ephemeral state, distributed rate limiting, background job
  workers, horizontal API scaling, production monitoring and load certification.
- Real outbound family messaging and autonomous unapproved writes.

## Recommended next production phase

1. Improve PHONE/entity recall and strict JSON reliability, then pass both the
   raw-input direct-model suite and deployed Gateway suite plus independent
   human/adversarial review.
2. Implement authenticated centre tenancy and role-based authorization at every
   database and Tool boundary.
3. Add Redis for shared cache/ephemeral coordination and a queue worker for
   ingestion, exports, Drive upload and other long-running jobs.
4. Add secrets management, encrypted sensitive storage, audit retention,
   backups, observability, cost limits and concurrent load tests.

This file is the release snapshot. Architecture rationale and operational detail
remain in `architecture.md`, `api-and-operations.md`, `tool-architecture.md`,
`rag-system.md`, `agent-evaluation.md`, and `local-safety-gateway.md`.
