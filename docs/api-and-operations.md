# API and local operation

## Start the application

```bash
source .venv/bin/activate
docker compose up -d postgres
alembic upgrade head
uvicorn app.main:app --reload
```

- Web workspace: `http://127.0.0.1:8000/`
- OpenAPI documentation: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Stop the local server with `Control-C` in the terminal that started it.

## Minimal HTTP example

Create a session:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{}'
```

Send a message using the returned `session_id`:

```bash
curl -X POST \
  http://127.0.0.1:8000/sessions/SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the EYLF say about play-based learning?",
    "request_id": "terminal-live-001"
  }'
```

Follow events as they arrive:

```bash
curl -N \
  "http://127.0.0.1:8000/sessions/SESSION_ID/events?request_id=terminal-live-001"
```

Fetch the public result:

```bash
curl \
  "http://127.0.0.1:8000/sessions/SESSION_ID/drafts/terminal-live-001"
```

New Main ReAct requests are draft-only. They never wait for approval or write
business records. The former approval route is no longer exposed.

## SSE behavior

The event stream sends named SSE frames such as `run_started`, `trace`,
`draft_ready`, `completed`, and `failed`. Each stored event
has a monotonic sequence number. A reconnecting client supplies
`after_sequence` and receives only newer records.

The connection closes when a run completes, fails, or is cancelled.

This is event streaming rather than token streaming. The UI can show truthful
workflow progress and recover after disconnects, while the final answer is
loaded from the durable draft endpoint.

## Local storage

The API, business records, and LangGraph checkpoints use the self-hosted
PostgreSQL container. Its port is bound only to `127.0.0.1`, and its durable
data lives in the Docker named volume `eduflow_postgres_data`. Chroma remains
under `data/chroma/`. Production has no SQLite fallback. Do not commit local
state or use real child or family information in development.

Useful database commands:

```bash
docker compose ps
alembic current
alembic check
docker compose stop postgres
```

## Common checks

```bash
python -m pytest
python scripts/run_evals.py
python scripts/run_reliability_checks.py
```

Live-model evaluation uses configured credentials and may produce variable
results. The deterministic test and reliability suites should remain stable.
