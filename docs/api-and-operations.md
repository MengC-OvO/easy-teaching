# API and local operation

## Start the application

```powershell
.\.venv\Scripts\Activate.ps1
docker compose up -d postgres
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\python.exe scripts\run_api.py
```

On macOS/Linux, activate with `source .venv/bin/activate` and use the same
`python scripts/run_api.py` wrapper. It selects the Windows event-loop policy
needed by psycopg and uses normal asyncio elsewhere.

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

## Google Drive MCP (optional)

The project integrates the open-source
[`taylorwilsdon/google_workspace_mcp`](https://github.com/taylorwilsdon/google_workspace_mcp)
server and exposes only two narrow capabilities to Main:

- `search_google_drive` searches the authorised teacher account and is read-only.
- `upload_export_to_google_drive` uploads only a managed `record_export` and
  always pauses for teacher approval. It cannot accept an arbitrary local path.

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

## SSE behavior

The event stream sends named SSE frames such as `run_started`, `trace`,
`draft_ready`, `approval_required`, `completed`, and `failed`. Each stored event
has a monotonic sequence number. A reconnecting client supplies
`after_sequence` and receives only newer records.

The connection closes when an Agent run completes, fails, or is cancelled.

This is event streaming rather than token streaming. The UI can show truthful
Agent progress and recover after disconnects, while the final answer is
loaded from the durable draft endpoint.

## Local storage

The API, business records, and LangGraph checkpoints use the self-hosted
PostgreSQL container. Its port is bound only to `127.0.0.1`, and its durable
data lives in the Docker named volume `easyteaching_postgres_data`. Chroma remains
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
