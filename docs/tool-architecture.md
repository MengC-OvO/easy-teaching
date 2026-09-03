# Tool and Worker architecture

## Main execution rule

Main ReAct owns the teacher-facing answer. It can call one tool, call several
independent `parallel_safe` tools together, or launch exactly two independent
deep Worker tasks. Dependent work is sequential: Main receives the first
Observation, then decides the next call. Model-generated dependency labels are
not part of the schema.

A single deep task stays in Main. Workers are used only when at least two tasks
each need multiple research steps and can run without one another's result.

## Registered tools

| Tool | Responsibility | Information source | Cost and permission |
| --- | --- | --- | --- |
| `get_class_context` | Current authorised class, focus, optional pseudonymous child IDs, relevant class memory | PostgreSQL | Local read, auto |
| `retrieve_knowledge` | EYLF/NQS/centre-policy evidence with hard source scope | SQLite FTS5 + Chroma; local Cross-encoder in deep mode | Standard uses one query embedding; deep also uses one query-rewrite model call, auto |
| `query_records` | Search authorised observations and educational records | PostgreSQL | Local read, auto |
| `read_draft_artifact` | Resolve one not-yet-saved generated draft by trusted request ID | PostgreSQL conversation results | Local read, auto |
| `get_daily_context` | Weather plus NSW public-holiday context for the centre | Open-Meteo + versioned local JSON calendar | External read, auto; only configured centre coordinates leave the system |
| `check_activity_safety` | Deterministic activity hazards and centre-rule checks | Local code/rules | Local read, auto |
| `save_observation` | Persist an objective observation after review | PostgreSQL | Controlled write, approval required |
| `save_educational_record` | Persist a learning story, analysis, plan, reflection or follow-up | PostgreSQL | Controlled write, approval required |
| `export_records` | Produce DOCX/PDF from already saved records | PostgreSQL + fixed local templates | Controlled write, approval required |
| `drive_operation` | Lazily discover Drive MCP tools, return their schemas to Main, then execute the selected remote tool through the same gateway | Optional Google Workspace MCP | Discovery/read-only calls auto; writes dynamically require approval; destructive calls forbidden |
| `read_uploaded_document` | Extract bounded text from a document uploaded in the current session | Scoped local file store | Local read, auto |
| `ingest_uploaded_document` | Add an uploaded centre document to isolated teacher/class knowledge indexes | Tenant-local JSONL + SQLite FTS5 + Chroma using local SentenceTransformer embeddings | Controlled write, approval required; no embedding egress |
| `search_official_web` | Search current government/ACECQA guidance on an allowlist | Optional Google Programmable Search | External read, auto |
| `transcribe_voice_note` | Convert a current-session audio note to text without saving a record | Optional local faster-whisper | Local read, auto |

Teacher preferences are injected by the context manager on every model turn;
they are not a tool. A new observation does not trigger knowledge or prior-record
retrieval unless the teacher requests alignment or the document depends on it.
The configured centre stores coordinates, so normal weather lookup needs one forecast
request rather than repeating a geocoding request.

## Workers

- `curriculum_research_worker`: only `retrieve_knowledge`,
  `search_official_web` and `check_activity_safety`.
- `record_context_worker`: only `get_class_context` and `query_records`.

Workers cannot write, approve, export, send externally or produce the final
teacher response. Each has a three-step bounded ReAct loop and receives a task
plus at least two explicit research questions.

## Controlled-write flow

1. Main organises the teacher's fragments, separates observation from
   interpretation, asks only for critical missing information, and forms exact
   validated fields.
2. The graph stores one frozen action with an argument hash and expiry, returns
   its preview, and stops at `waiting_for_approval`.
3. Approval atomically claims that action. Concurrent/repeated approvals cannot
   execute the side effect twice.
4. The registered tool executes the frozen arguments in the trusted
   teacher/class scope; the result and audit event are persisted.

## PostgreSQL domain tables

`centres`, `teachers`, `classes`, `teacher_class_memberships` and `children`
define authorisation scope. `observations` and `observation_children` keep raw
objective events separate from `educational_records` and their links.
`record_exports`, `tool_action_requests`, `knowledge_sources` and `audit_events`
cover exports, approvals, index provenance and traceability. Long-term memory
retains teacher/class ownership plus confidence and review metadata.

## Verification

Offline and zero-provider-cost:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\alembic.exe upgrade head --sql
```

After PostgreSQL is running and migrations are applied, the user-run live suite
can deliberately consume configured provider quota:

```powershell
.\.venv\Scripts\python.exe scripts\test_phase1_live.py --scenarios class_context
.\.venv\Scripts\python.exe scripts\test_phase1_live.py --scenarios class_context eylf_rag observation_approval
```

The second command may call both the chat and embedding APIs. The script never
auto-approves a write.
