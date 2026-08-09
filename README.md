# EduFlow AU Agent

EduFlow AU Agent is a safety-aware teacher workflow assistant for Australian
early childhood education. It combines FastAPI, LangGraph, controlled tools,
local memory, and retrieval-augmented generation (RAG) to turn educator
requests into reviewable drafts and evidence-backed answers.

The repository is both a working local application and a learning project. All
example data must be synthetic, public, or thoroughly de-identified.

## What it does

- **One Main ReAct assistant** — handles planning, policy, documentation and
  family-draft requests without routing into fixed top-level specialists.
- **Activity planning drafts** — combines class context, EYLF evidence, safety
  checks and optional public context inside the same bounded loop.
- **Learning record drafts** — creates reviewable text only; saving is disabled
  in the current production graph.
- **Policy question answering with citations** — reuses the existing hybrid RAG
  evidence and source metadata.
- **Family communication drafts** — prepares reviewable wording without sending
  a real message.
- **Controlled research** — chooses one Tool, a concurrent Tool batch, or a
  bounded Worker batch for the current turn only.
- **Evidence-backed drafts** — reuses EYLF/policy RAG, citations, safety checks,
  scoped class context and public weather/resource tools.
- **Conversation continuity** — maintains LangGraph checkpoints, compact
  short-term context, and scoped long-term teacher/class memory.
- **Web workspace** — offers a ChatGPT-style local interface for sessions,
  messages, SSE progress events, draft display, and citations.

## System overview

```mermaid
flowchart LR
    Browser["Teacher web workspace"] --> API["FastAPI API"]
    API --> Runtime["Application runtime"]
    Runtime --> Graph["LangGraph main graph"]
    Graph --> Main["Main ReAct loop"]
    Main --> Validate["Code validation"]
    Validate --> Tools["Single / concurrent Tool"]
    Validate --> Workers["Bounded Worker fan-out"]
    Tools --> Main
    Workers --> Main
    Tools --> Knowledge["Hybrid RAG + citations"]
    Graph --> Store[("Self-hosted PostgreSQL")]
    API --> Events["Persisted SSE events"]
    Events --> Browser
```

One application-scoped runtime owns the SQLAlchemy store, PostgreSQL
checkpointer, and compiled graph. HTTP routes accept work and return quickly;
FastAPI then runs the accepted request asynchronously, persists its public
outcome, and publishes ordered events that the browser can replay over
Server-Sent Events (SSE).

The Main assistant does not create a complete future plan. On each loop it
chooses only the next executable action: one Tool/MCP call, a concurrent batch
of independent Tools, a bounded fan-out of independent Workers, a clarification,
or the final draft. Code validates every decision before anything executes.

For a deeper walkthrough, see [Architecture](docs/architecture.md) and
[API and local operation](docs/api-and-operations.md). Important corrected
behaviors are recorded in [Engineering decisions](docs/engineering-decisions.md).

## Repository structure

```text
app/
  agents/       Main ReAct, decision validation, execution and Worker profiles
  api/          HTTP routes, runtime composition, execution and recovery
  schemas/      validated HTTP, graph, decision and observation contracts
  services/     persistence, models, retrieval, retries and domain services
  tools/        controlled tool definitions, permissions and handlers
  web/          local browser interface (HTML, CSS and JavaScript)
  workflows/    production Main ReAct graph and async PostgreSQL checkpointing

data/
  knowledge/    tracked source manifest and processed public knowledge
  evals/        deterministic evaluation and reliability manifests
  chroma/       ignored local vector index

docs/           stable architecture and operating documentation
evals/          offline evaluation models, evaluators and runner
scripts/        local demos, ingestion, evaluation and maintenance entry points
tests/          unit and integration coverage
```

This structure keeps delivery code under `app/`, offline quality measurement
under `evals/`, local operator commands under `scripts/`, and explanatory
material under `docs/`. The core modules were not moved merely for appearance;
their current imports already express useful boundaries.

## Quick start

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your local model and embedding credentials to `.env`. Never commit that
file. Replace the PostgreSQL password placeholder in `POSTGRES_PASSWORD`,
`DATABASE_URL`, and `CHECKPOINT_DATABASE_URL` with the same long random value.

Start the self-hosted local PostgreSQL service and apply schema migrations:

```bash
docker compose up -d postgres
alembic upgrade head
```

Start the application:

```bash
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is
available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and the
health endpoint is `GET /health`.

The browser page uses the existing API; it does not bypass LangGraph or return
mock answers. It creates a durable session, submits a message, follows persisted
SSE workflow events and fetches the resulting draft. The production graph is
draft-only and exposes no approval, formal-write, or external-send route.

PostgreSQL is bound only to `127.0.0.1` in local development. SQLAlchemy owns
EduFlow operational and business tables; LangGraph's official `PostgresSaver`
owns its checkpoint tables in the same database. Chroma remains the separate
RAG vector store.

### Optional Supabase login

The local app remains authentication-free by default. To enable real email and
password login, create a free Supabase project, add or invite a user under
Authentication, and set these values in `.env`:

```bash
AUTH_ENABLED=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-publishable-key
```

Restart the app and open the web workspace. Supabase validates the password;
FastAPI exchanges the resulting access token for an HTTP-only login cookie and
uses the trusted Supabase user ID as `teacher_id`. Authenticated users can only
open, message, stream, or read sessions that they own. Never put a
Supabase secret or service-role key in browser configuration.

## Knowledge setup

Prepare tracked source documents as chunks:

```bash
python scripts/ingest_knowledge.py \
  --output data/knowledge/processed/chunks.jsonl
```

Build the local Chroma index:

```bash
python scripts/build_vector_index.py --reset --batch-size 5
```

Query it directly:

```bash
python scripts/query_vector_index.py \
  "What does the EYLF say about play-based learning?" \
  --mode hybrid --top-k 5
```

The current retrieval stack supports dense, BM25, and hybrid search with an
optional cross-encoder reranker. Gemini creates 768-dimensional embeddings;
Chroma stores and searches them using cosine distance.

## API flow

The public workflow is deliberately small:

1. `POST /sessions` creates a conversation and LangGraph thread.
2. `POST /sessions/{session_id}/messages` accepts one idempotent request.
3. `GET /sessions/{session_id}/events?request_id=...` streams ordered progress
   events using SSE.
4. `GET /sessions/{session_id}/drafts/{request_id}` returns the public draft
   and citations.

SSE currently streams workflow lifecycle events, not individual LLM tokens.
That distinction keeps replay and reconnect behavior deterministic: events are
stored first, then delivered from a sequence cursor.

## Safety boundaries

EduFlow is a teacher assistant. It **must not**:

- Diagnose children
- Provide medical advice
- Provide legal compliance conclusions
- Send real messages to families
- expose raw real child or family private information to a model

| Risk | Typical capability | Behavior |
| --- | --- | --- |
| L0 read-only | Search policy or read synthetic class context | Execute and record evidence |
| L1 draft | Generate an activity, record, or communication draft | Mark clearly as a draft |
| L2 controlled write | Save or overwrite a record | Require scoped teacher approval |
| L3 forbidden or handoff | Sending, diagnosis, medical/legal judgment, raw PII | Refuse or hand off |

The current production graph never performs a controlled write or real-world
side effect. A model never receives arbitrary Python functions: Tool and Worker
names are registered in code with validated schemas, risk levels, allowlists,
timeouts, dependency checks and trusted teacher/class execution scope.
Human approval is required before any future controlled write is re-enabled;
the current draft-only graph never reaches that boundary.

## Testing and evaluation

Run the complete automated suite:

```bash
python -m pytest
```

Run the 30-case agent evaluation:

```bash
python scripts/run_evals.py
python scripts/run_evals.py --live-model
```

Run deterministic reliability and failure-injection checks:

```bash
python scripts/run_reliability_checks.py
```

Run one question through the real configured model and the production
PostgreSQL-backed Main ReAct path:

```bash
python scripts/ask_live.py \
  "What does the EYLF say about play-based learning?" \
  --trace
```

This command persists a synthetic terminal-demo session in the local database
and prints the final draft, citations, and optional execution trace.

Latest local verification after the async refactor:

| Check | Result |
| --- | --- |
| Complete pytest suite | 230 passed |
| Reliability matrix | 14/14 passed |
| Offline agent evaluation | 28/30 passed |
| Python compilation and diff whitespace check | Passed |

The two offline evaluation misses are known RAG source-selection cases, not
Main ReAct routing or execution failures.

The evaluation suite retains component-level routing checks and now exercises
the production Main ReAct trajectory for single Tool, concurrent Tool,
dependency, parallel Worker and clarification paths. Reliability scenarios
exercise retryable model failures, non-retryable failures, structured-output
repair, fallback responses, and observable API failure events. Evaluation data
is development-only and is not part of the production API response path.

## Current status and known gaps

Completed work includes the unified Main ReAct production graph, controlled
single/concurrent tools, bounded Worker fan-out/fan-in, hybrid RAG, checkpointed
context and memory, draft-only execution, idempotent FastAPI routes, SSE replay,
offline evaluation, retries, fallbacks, and fault validation. The former
Specialist, Skill and approval execution paths have been removed. Historical
database columns remain until an explicit, data-safe Alembic migration removes
them.

Known follow-up work:

- improve RAG recall and source-selection quality for the remaining evaluation
  misses;
- add stronger privacy-session handling for real multi-turn learning records;
- run a real PostgreSQL migration and API smoke test when Docker is available;
- remove historical database compatibility columns only through a backed-up,
  reversible Alembic migration;
- add true token streaming only if the model provider and product experience
  justify the added complexity;
- keep WebSocket support out until bidirectional real-time interaction is
  actually needed.

## Design principles

- **Validated boundaries:** Pydantic contracts guard HTTP, graph, Main decision,
  Worker observation, Tool, and evaluation interfaces.
- **Least privilege:** Main sees registered read-only capabilities; each Worker
  receives a smaller domain-specific allowlist and step budget.
- **Draft-only boundary:** the production graph cannot save, send, approve or
  perform another side effect.
- **Replayable execution:** checkpointed graph state and persisted SSE events
  support safe recovery and reconnection.
- **Inspectable quality:** evaluations report per-capability results instead of
  hiding failures inside one aggregate score.

## License

The source code is available under the [MIT License](LICENSE).

The license does not make the example application production-ready. Any real
deployment still requires privacy review, security controls, a data-retention
policy, and organisation-specific governance. Use only synthetic, public, or
thoroughly de-identified education data while developing locally.
