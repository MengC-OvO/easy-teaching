# API and local operation

## Start the application

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

On macOS/Linux, activate with `source .venv/bin/activate` and use the same
`python scripts/run_api.py` wrapper. It selects the Windows event-loop policy
needed by psycopg and uses normal asyncio elsewhere.

- Web workspace: `http://127.0.0.1:8000/`
- OpenAPI documentation: `http://127.0.0.1:8000/docs`
- Liveness: `http://127.0.0.1:8000/health`
- PostgreSQL/Redis readiness: `http://127.0.0.1:8000/ready`

Stop the local server with `Control-C` in the terminal that started it.

## Background execution and delivery guarantees

FastAPI does not run LangGraph inside `BackgroundTasks`. The message endpoint
commits the run and a redacted execution payload to PostgreSQL in one
transaction, then an Outbox relay publishes only the `request_id` to Celery via
Redis. If broker publication fails after admission, accepted work is delayed
instead of lost. When the Redis-backed admission limiter itself is unavailable,
new requests fail closed with HTTP 503 and no run is created.

Celery workers use late acknowledgement, worker-loss redelivery, a prefetch
multiplier of one, soft/hard time limits and bounded exponential-backoff retries.
PostgreSQL leases suppress concurrent duplicate deliveries. Exactly-once network
delivery is not assumed: request IDs, session uniqueness, frozen-action hashes
and tool idempotency keys make repeated delivery safe. PostgreSQL remains the
source of truth for run state and results; Celery's result backend is disabled.

Redis also provides an atomic per-user/IP fixed-window limit for new message
requests. Idempotent replays of an existing `request_id` return the stored run
without consuming another limit slot.

Operational commands:

```bash
docker compose ps
docker compose logs -f api worker redis
docker compose up -d --scale worker=3
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping
python scripts/test_redis_stream_live.py --events 200 --concurrency 20
```

On Windows, run Celery in the Linux container; Celery does not officially
support native Windows workers. `TASK_EXECUTION_MODE=inline` exists only for
deterministic tests and local debugging.

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

Ordinary generation returns a draft. When the teacher explicitly asks to save
an observation, educational record or export, the result may have status
`waiting_for_approval` and contains the frozen fields. Approve or reject that
exact action with:

```bash
curl -X POST \
  "http://127.0.0.1:8000/sessions/SESSION_ID/approvals" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"REQUEST_ID","decision":"approve"}'
```

Approval cannot replace the frozen arguments. External sending is not exposed.

## Uploaded documents and voice notes

The web composer accepts PDF, DOCX, TXT, Markdown, CSV and common audio formats.
The equivalent HTTP request is:

```bash
curl -X POST "http://127.0.0.1:8000/sessions/SESSION_ID/uploads" \
  -F "file=@centre-policy.pdf"
```

The response contains an opaque `file_id`. Include that ID in the teacher
message so Main can call `read_uploaded_document`, request approval for
`ingest_uploaded_document`, or call `transcribe_voice_note`. Files remain bound
to their originating teacher, class and session. Approved centre knowledge uses
isolated local BM25 and Chroma indexes. Its Dense vectors come from the configured
local SentenceTransformer, so document text is not sent to the external Gemini
embedding provider.

Official web search and local transcription are optional. Configure
`OFFICIAL_WEB_SEARCH_*` to register the allowlisted search Tool. Install
`requirements-transcription.txt` and enable `VOICE_TRANSCRIPTION_ENABLED` to
register local faster-whisper transcription.

## Google Drive MCP (optional)

The project integrates the open-source
[`taylorwilsdon/google_workspace_mcp`](https://github.com/taylorwilsdon/google_workspace_mcp)
server through one registered `drive_operation` gateway. Main first calls
`action=discover`; the gateway lazily requests `tools/list` and returns the remote
tool names, model-facing schemas, and locally resolved risk. On the next ReAct step
Main calls the same gateway with `action=execute`, `tool_name`, and schema-valid
arguments. Read-only calls execute automatically, writes pause for teacher approval,
and destructive calls are forbidden. `create_drive_file` receives a safe
`export_id` schema instead of arbitrary local paths or file bytes. Execution uses
the catalog captured by the preceding discovery, so an undiscovered or changed
remote tool cannot bypass the permission decision made for that step.

The server and Google APIs do not charge an MCP fee. Google still requires a
one-time OAuth client and browser consent before Drive data can be accessed.
Create a Desktop OAuth client in a Google Cloud project, enable the Drive API,
then add these values to the untracked `.env` file:

```dotenv
GOOGLE_DRIVE_MCP_ENABLED=true
GOOGLE_DRIVE_USER_EMAIL=teacher@example.com
GOOGLE_OAUTH_CLIENT_ID=your-client-id
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
WORKSPACE_MCP_CREDENTIALS_DIR=data/local/google_workspace_mcp
```

On the first Drive request, complete Google's browser consent. Tokens remain in
the ignored local credentials directory. The application starts the MCP server
over stdio with only the Drive/core tool set; no separate Docker service or paid
MCP host is required.

To verify the installed server without opening OAuth or calling any Google Drive
API, run the read-only catalog smoke test from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\test_google_drive_mcp.py --list-tools-only
```

## SSE behavior

The event stream sends named SSE frames such as `run_started`, `trace`,
`draft_ready`, `approval_required`, `completed`, and `failed`. Node-level
progress uses a capped Redis Stream with a one-hour TTL. `XREAD BLOCK` waits for
new records without repeatedly querying PostgreSQL. A reconnecting browser
supplies `after_event_id` (or the standard `Last-Event-ID`) and resumes after the
last Redis record it received.

PostgreSQL stores the run, final draft, approval, citations, checkpoints and
important lifecycle events. It does not store ordinary node progress in Celery
mode. If Redis progress is unavailable, SSE degrades to a one-second durable
status check; the Agent run itself continues and its final result remains safe.

The connection closes when an Agent run completes, fails, or is cancelled.

This is node-event streaming rather than token streaming. Redis contains only
allowlisted step names and generic messages, never prompts, model output or Tool
arguments. The final answer is always loaded from the durable draft endpoint.

## Local storage

The API, business records, Outbox, and LangGraph checkpoints use the self-hosted
PostgreSQL container. Redis uses AOF persistence for queued task transport, but
is not the business source of truth. Both ports are bound only to `127.0.0.1`,
and durable data lives in Docker named volumes. Chroma remains
under `data/chroma/`. Production has no SQLite fallback. Do not commit local
state or use real child or family information in development.

Useful database commands:

```bash
docker compose ps
alembic current
alembic check
docker compose stop api worker redis postgres
```

## Common checks

```bash
python -m pytest
python scripts/run_evals.py
python scripts/run_reliability_checks.py
```

Live-model evaluation uses configured credentials and may produce variable
results. The deterministic test and reliability suites should remain stable.
