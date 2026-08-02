# API and local operation

## Start the application

```bash
source .venv/bin/activate
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

If it is waiting for approval:

```bash
curl -X POST \
  http://127.0.0.1:8000/sessions/SESSION_ID/approvals \
  -H "Content-Type: application/json" \
  -d '{"request_id":"terminal-live-001","decision":"approve"}'
```

## SSE behavior

The event stream sends named SSE frames such as `run_started`, `trace`,
`draft_ready`, `approval_required`, `completed`, and `failed`. Each stored event
has a monotonic sequence number. A reconnecting client supplies
`after_sequence` and receives only newer records.

The connection closes when a run completes, fails, is cancelled, or pauses for
approval. After submitting an approval decision, the client opens a new stream
to follow the resumed graph.

This is event streaming rather than token streaming. The UI can show truthful
workflow progress and recover after disconnects, while the final answer is
loaded from the durable draft endpoint.

## Local storage

The application creates ignored files under `data/local/` and `data/chroma/`.
These may contain local development state. Do not commit them or use real child
or family information.

## Common checks

```bash
python -m pytest
python scripts/run_evals.py
python scripts/run_reliability_checks.py
```

Live-model evaluation uses configured credentials and may produce variable
results. The deterministic test and reliability suites should remain stable.
