# Outbox ownership and recovery

Approved controlled writes now reuse this delivery/lease mechanism through an
`approved_action` payload phase with atomic consent admission, record
transactions and external-effect boundaries.

## Deployment

Stop/drain all old API relays and Celery workers before deploying this change.
Run `python -m alembic upgrade head` using the intended deployment environment,
then start the new API and workers. Migration `0005_outbox_lease_token` adds the
nullable `conversation_task_outbox.lease_token` column. Old binaries do not
enforce ownership and must not run alongside the new version. This change does
not automatically apply migrations to a running database.

Existing expired leases without tokens are recovered by the relay. Fresh legacy
leases must expire after old workers are stopped. Do not manually clear a live
worker's lease.

## Three fixes

1. A worker may claim `publishing` as well as `published`: receipt of the broker
   message is enough to attempt an atomic database claim. It replaces the lease
   token. The publisher's late success or failure callback can no longer reset
   the worker's `running` state. Database claim locks still suppress duplicate
   consumers.
2. Every relay cycle reconciles up to 100 stranded tasks under row locks before
   publishing. `published` rows older than `OUTBOX_STALE_PUBLISHED_SECONDS`
   (default 1200 seconds) and expired `running`/`publishing` leases are eligible.
   Terminal Runs are reconciled to completed tasks; other eligible rows become
   pending and are republished. Fresh queued tasks and live execution leases
   are not changed. Compensation is bounded per cycle and requires a live relay.
   A long healthy backlog can cause duplicate messages; tune the threshold for
   queue latency. Consumer claims and business idempotency remain necessary.
3. Every publisher/executor claim gets a new UUID token. Finish/release requires
   the matching token, expected status, and an unexpired lease. Stale callbacks
   return false; stale workers cannot release another worker's retry lease.
   Worker ownership is propagated with a ContextVar, not GraphState. Run status,
   result, lifecycle event, memory, action creation and record-save transactions
   check and lock the owning task before writing. Non-worker API operations do
   not acquire an execution lease.

## Boundaries

This is not exactly-once execution. Redis already has AOF/everysec enabled in
compose; durable Outbox reconciliation is still needed for message loss. Leases
have no heartbeat renewal; execution limits must remain below lease duration.
Task tokens do not cancel an old process or fence a remote API, filesystem writes,
privacy mapping consumption, or the independent LangGraph checkpointer. Such
operations still need idempotency/fencing at their own boundary. New business
write paths must explicitly use the ownership guard.

## Verification

`tests/test_outbox_leases.py` exercises SQL state predicates, early consumption,
late publisher callbacks, expired ownership, compensation and protected business
writes using a synchronous SQLite test adapter. SQLite does not verify PostgreSQL
row locking. `test_real_postgres_outbox_takeover_and_stale_publisher_fencing` in
`tests/test_postgres_redis_integration.py` exercises ten concurrent PostgreSQL
claims when the opt-in integration service URLs are configured and the migration
has been applied to an isolated test database. Never point integration tests at
production: some existing integration tests clear their configured broker queue.
