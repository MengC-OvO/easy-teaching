# Live progress and checked answer delivery

The local `inline` message route acknowledges the request before executing its
FastAPI background task. This is a development mode; Celery/Outbox remains the
production queue. Both use graph `astream` updates. Inline mode persists small
progress records as nodes finish; Celery publishes them through Redis. Progress
uses known tool labels and result status, never raw arguments or source text.

The browser shows current work and expandable history. Final answers are delivered
by `/sessions/{session_id}/drafts/{request_id}/stream`, reusing the existing draft
authorization and readiness checks. Only a persisted, privacy-restored public
draft is streamed. This is **delivery streaming after validation**, not raw model
token streaming. The stream sends metadata, Unicode text deltas and a completion
event; character offsets support Last-Event-ID reconnection. The browser falls
back to the authoritative full draft after repeated stream failures. Both SSE
and fallback text use the browser's paced rendering queue (50 Unicode code points
per second), so a network burst or completion event cannot flush the entire
answer immediately. The script URL is versioned to refresh stale browser caches. Answer text
is escaped before rendering a small Markdown subset. No second model generates
the streamed text, and no answer text is placed in Redis progress records.

For activity drafts, Main submits a complete final candidate. The validator
preserves it while checking its exact content. A passed matching check publishes
that candidate without another Main call. Failed checks return to Main; unresolved
controlled-write requirements prevent the direct-final path. New runs reset the
candidate. Existing approval and evidence gates remain in place.

Weather results have a bounded per-tool-runtime cache (128 entries, ten-minute
TTL), keyed by trusted centre/location/timezone and date. Authorization/location
lookup precedes cache access. Errors are not cached; output includes cache_hit and
fetched_at. This does not share results between API/worker processes and does not
remove the model's tool-selection turn. Missing target_date means today in the
centre timezone. Main receives a current UTC timestamp and is instructed to omit
the date for today's weather, and to batch independent class/weather reads.

Relevant regression coverage: `tests/test_live_answer_delivery.py`, plus existing
API, graph, Redis progress, safety, approval and retrieval suites.
