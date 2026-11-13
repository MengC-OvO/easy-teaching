# EasyTeaching

A workspace for early childhood educators to plan activities, prepare records and find policy guidance with source references.

## Workspaces

- **Activity planning drafts** informed by EYLF, NQS and authorized centre material.
- **Observation and educational records** with teacher review before saving.
- **Policy question answering with citations** and source-scope filtering.
- **Family communication drafts** prepared for educator review.
- Account-based access, authorized class selection, document uploads and resumable conversations.

## Deployment

See [Deployment and operations](docs/deployment.md) for configuration, TLS, account provisioning, indexing, backups, upgrades and acceptance checks.

The current release is for synthetic or de-identified information. Identifiable child and family data are outside the release scope. The optional privacy gateway has not passed its release gate.

Requirements: Python 3.12, Docker with Compose, and chat/embedding provider credentials. Production also requires Supabase authentication and a public HTTPS origin.

```bash
cp .env.example .env
# Configure credentials before starting services.
docker compose build
docker compose up -d postgres redis
docker compose run --rm migrate
# Uses embedding provider quota; resume avoids re-indexing completed chunks.
docker compose run --rm --no-deps api python scripts/build_vector_index.py --resume --batch-size 5
docker compose up -d api worker
```

On PowerShell, use `Copy-Item .env.example .env`. The local workspace is available at <http://127.0.0.1:8000>. `/health` checks API liveness; `/ready` checks PostgreSQL and Redis. Local API documentation is at `/docs`.

Production requires `APP_ENV=production`, `AUTH_ENABLED=true`, `PUBLIC_ORIGIN`, configured credentials, Redis rate limiting and Celery workers. Empty knowledge indexes prevent production startup. The production guide includes a TLS reverse-proxy configuration.

## Teacher review and access

| Level | Operation | Behavior |
| --- | --- | --- |
| L0 read-only | Authorized records and policy sources | Scoped retrieval |
| L1 draft | Activity, record or communication draft | Teacher review |
| L2 controlled write | Save, export or upload | Frozen arguments and explicit approval |
| L3 forbidden or handoff | Diagnosis, medical/legal conclusions, sending | Refuse or hand off |

**Human approval is required** for each controlled write. EasyTeaching must not **Diagnose children**, provide **medical advice** or **legal compliance conclusions**, or **Send real messages to families**. External message sending is not implemented.

## Development and verification

```bash
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install -r requirements.lock pip-audit
python -m pip check
python -m pytest -q
node tests/test_answer_render_queue.cjs
node tests/test_workspace_browser.cjs
python scripts/run_evals.py --min-pass-rate 1
python scripts/audit_dependencies.py
```

GitHub Actions runs regression, browser behavior and PostgreSQL/Redis integration checks, dependency auditing and the Docker build. See the deployment guide for the narrowly reviewed embedded-Chroma audit exceptions. Live provider, capacity and restore tests remain environment-specific release checks.

## Technical documentation

- [API and operations](docs/api-and-operations.md)
- [Tool and worker architecture](docs/tool-architecture.md)
- [Retrieval and indexing](docs/rag-system.md)
- [Evaluation methods](docs/agent-evaluation.md)
- [Optional privacy gateway](docs/local-safety-gateway.md)

## License

[MIT](LICENSE). Bundled education documents retain their original ownership and terms.
