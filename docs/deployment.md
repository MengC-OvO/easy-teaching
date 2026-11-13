# Deployment and operations

This release supports authenticated educator workspaces using synthetic or de-identified information. It does not enable the use of identifiable child or family records. The optional privacy gateway has not passed its release gate.

## Prerequisites

- A Linux x86_64 server with Docker Engine and Compose, persistent storage, and enough memory for the selected local embedding/reranking models. The deployment image uses Python 3.12 and CPU inference. Size the server with your workload before accepting users.
- A DNS name pointing to the server. Permit inbound ports 80 and 443; do not expose PostgreSQL, Redis or Chroma.
- A Supabase project with educator accounts created by an administrator. Disable public signups for an invitation-only deployment. Use the publishable key, never a service-role key, in `SUPABASE_PUBLISHABLE_KEY`.
- Chat and embedding service credentials. Optional Drive, web search and transcription remain disabled unless configured and tested.

## Configure

Copy `.env.example` to `.env`. Set:

```dotenv
APP_ENV=production
AUTH_ENABLED=true
PUBLIC_ORIGIN=https://teaching.example.org
SITE_HOST=teaching.example.org
SUPABASE_URL=https://PROJECT.supabase.co
SUPABASE_PUBLISHABLE_KEY=YOUR_PUBLISHABLE_KEY
TASK_EXECUTION_MODE=celery
API_RATE_LIMIT_ENABLED=true
```

Replace all database, Redis, chat and embedding placeholders. Generate independent random passwords; use hexadecimal passwords in Compose connection URLs to avoid URL delimiter escaping errors. Keep `.env` outside version control, restrict server file permissions, and use your platform's secret manager where available. `PUBLIC_ORIGIN` must exactly match the browser origin, including any non-default port. The API refuses incomplete production configuration.

## Build and initialize

Run from the project root:

```bash
docker compose build
docker compose up -d postgres redis
docker compose run --rm migrate
# Builds the shared public vector index; this consumes embedding provider quota.
docker compose run --rm --no-deps api python scripts/build_vector_index.py --resume --batch-size 5
```

The lexical index is built from the bundled public corpus during image construction. The vector index persists in a shared Docker volume used by the API and workers. Production startup rejects empty knowledge indexes. Do not rebuild an index while serving traffic: stop API and workers first, back up the volume and use the same image/model settings for indexing and serving.

Production does not seed example teachers or children. Provision an educator's Supabase user UUID and authorized class:

```bash
docker compose run --rm --no-deps api python scripts/provision_workspace.py \
  --user-id 00000000-0000-0000-0000-000000000001 \
  --teacher-name "Educator" --centre-id centre-01 --centre-name "Centre" \
  --state NSW --suburb Sydney --class-id room-01 --class-name "Preschool" --age-group 3-5
```

Replace the example UUID with the actual account ID. Repeating the same assignment is safe; existing cross-centre assignments or revoked memberships are rejected. General planning and policy questions work without a class assignment; scoped documents and records require one. Child roster administration is outside this release; no real child roster should be imported.

```bash
docker compose -f compose.yaml -f compose.production.yaml up -d
docker compose ps
curl --fail https://teaching.example.org/ready
```

Caddy obtains and renews TLS certificates. The API stays bound to loopback on the host, and Caddy reaches it through the private Compose network. The proxy streams progress without response buffering. API documentation is disabled in production.

## Acceptance checks

1. Confirm `/health` returns 200 and `/ready` returns 200. Readiness checks PostgreSQL and both Redis clients with bounded timeouts; it does not assert worker capacity or provider availability.
2. Sign in with two separate educator accounts. Confirm each sees only assigned classes and cannot open the other's session or uploaded file.
3. Select a class, upload a de-identified text document and request a draft. Verify ordered progress, final content and citations.
4. Request a controlled save, reject it, then repeat and approve it. Confirm no write occurs before approval and refresh/retry does not duplicate the write.
5. Restart a worker during a synthetic request and confirm outbox recovery. Monitor worker presence with `docker compose exec worker celery -A app.tasks.celery_app:celery_app inspect ping`.
6. Sign out and refresh. Confirm the account's workspace is inaccessible. Browser history is per account and per tab session; closing the tab discards the local recent list.

## Backups and upgrades

Back up PostgreSQL and the `easyteaching_local_data`, `easyteaching_exports` and `easyteaching_vector_data` volumes. Database and file backups must describe the same application state: pause new requests and wait for active workers to finish before taking a coordinated backup. Store encrypted backups off-host and test restoring to an isolated deployment. Redis AOF supports recovery but does not replace the PostgreSQL outbox or database backups. Configure retention for backups, conversation records and uploaded files according to your organization's approved policy.

Before upgrading, preserve the previous image and a verified backup. Stop API and workers, build the new image, run migrations once, rebuild indexes if required, then start services and perform acceptance checks. This release upgrades LangGraph/checkpoint dependencies; validate old conversation checkpoints on a restored copy before upgrading an existing deployment. For a rollback, restore the previous image together with its compatible database and file backup. Never delete volumes with `docker compose down -v` during routine upgrades.

## Verification and dependency policy

`requirements.txt` pins direct dependencies; `constraints-security.txt` specifies patched transitive minimums. `requirements.lock` is the universal Python 3.12 development resolution. Docker and Linux CI use `requirements-linux.lock`, which selects the official, SHA256-verified CPU PyTorch wheel to avoid downloading CUDA packages. Refresh with `uv pip compile --python-version 3.12 --universal requirements.txt -o requirements.lock` and `uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 requirements-linux.in -o requirements-linux.lock`, then rerun tests and the audit.

```bash
python -m pip install -r requirements.lock pip-audit
python -m pip check
python scripts/audit_dependencies.py
python -m pytest -q
node tests/test_answer_render_queue.cjs
node tests/test_workspace_browser.cjs
python scripts/run_evals.py --min-pass-rate 1
```

CI also runs PostgreSQL/Redis integration tests and builds the image. Chroma 1.5.9 has outstanding advisories for its HTTP collection-management and server authorization interfaces (`PYSEC-2026-311`, `PYSEC-2026-3813`, `PYSEC-2026-3814`, `PYSEC-2026-3815`). Those interfaces are not enabled: both public and scoped retrieval use embedded `PersistentClient` with application-controlled collection configuration. These are narrowly documented audit exceptions, not fixed vulnerabilities. Do not run `chroma run`, publish a Chroma port, accept user-defined embedding functions or enable remote model code. Reassess these exceptions if the architecture changes. The audit fails for other findings or unaudited packages.

The password login uses the [Supabase Auth token endpoint](https://supabase.com/docs/guides/auth/passwords); external JavaScript is not loaded. Session cookies are HttpOnly, Secure outside local/test mode, and SameSite=Lax. Write requests reject cross-origin browser requests. Production also requires a trusted reverse proxy, resource limits appropriate to the host, monitoring of readiness/worker failures, and a workload-specific capacity test before rollout.
