# EduFlow AU Agent

EduFlow AU is a learning project for building a teacher workflow agent for
Australian early childhood education scenarios. The project focuses on a
runnable, testable Agent application backbone rather than a complete commercial
product.

## Current Scope

Day 1 built the project foundation:

- Python backend project structure
- FastAPI health endpoint
- Pydantic graph state
- LangGraph StateGraph skeleton
- Scope and safety boundaries
- Initial tests

Day 2 adds the first controlled tool layer:

- ToolDefinition and ToolResult contracts
- ToolRegistry with parameter validation and permission checks
- SQLAlchemy-backed SQLite mock data store
- Mock tools for class profile lookup, policy index search, and draft saving

The current baseline includes the repository layout, a minimal FastAPI health
endpoint, Pydantic graph state, LangGraph skeleton, and executable mock tools.

## Product Scope

The agent will support synthetic teacher workflow scenarios:

- Activity planning drafts
- Learning record drafts
- Policy question answering with citations
- Family communication drafts

## Safety Boundaries

EduFlow AU is a teacher assistant. It must not:

- Diagnose children
- Provide medical advice
- Provide legal compliance conclusions
- Send real messages to families
- Use raw real child or family private information

All data in this project should be synthetic or thoroughly de-identified.

### Risk Levels

| Level | Typical action | System behavior |
| --- | --- | --- |
| L0 read-only | Search policy text, read synthetic class configuration | Execute automatically and record sources |
| L1 draft | Generate activity plans, learning records, or family message drafts | Execute automatically, clearly marked as Draft |
| L2 controlled write | Save, overwrite, or export records | Show the change and require teacher confirmation |
| L3 forbidden or handoff | Real sending, diagnosis, medical/legal judgment, raw PII | Refuse or hand off with a clear boundary explanation |

The model may draft and reason, but code-level validation controls what can be
saved, exported, or refused.

### Human Approval

Human approval is required before any controlled write or real-world side
effect. Approval is treated as a scoped authorization boundary, not as a casual
review step after unrestricted model behavior.

## Tool System

Tools are not plain functions exposed directly to the model. Each tool is
described by a `ToolDefinition` with:

- `name`
- `description`
- `category`
- Pydantic `input_model`
- Pydantic `output_model`
- `risk_level`
- `permission`
- `handler`

Tool execution returns a structured `ToolResult`:

- `success`
- `data`
- `error`
- `risk_level`
- `trace`

The `ToolRegistry` is the code-level boundary between model intent and tool
execution. It handles:

- duplicate tool-name protection
- tool lookup
- Pydantic argument validation
- approval checks
- forbidden-tool blocking
- handler exception wrapping
- structured `ToolResult` output

This means the model may request a tool, but it cannot bypass validation,
permission checks, risk classification, or handler-level execution rules.

## Mock Data Store

The Day 2 mock data layer uses SQLAlchemy with SQLite. SQLite stores data in a
local file, so no separate database server is required.

Default local database path:

```text
data/local/eduflow.sqlite3
```

This path is ignored by Git. The database contains synthetic data only.

Current SQLAlchemy models:

- `ClassProfile`
- `DraftRecord`
- `PolicyIndexEntry`

The service layer exposes `EduFlowStore`, which provides:

- `initialize()`
- `get_class_profile(class_id)`
- `search_policy_index(query)`
- `save_draft(...)`

Tool handlers call `EduFlowStore`; they do not write SQL directly.

## Mock Tools

The current mock tool registry registers three tools:

| Tool | Risk | Permission | Purpose |
| --- | --- | --- | --- |
| `get_class_profile` | L0 read-only | Auto execute | Read synthetic class profile data |
| `search_policy_index` | L0 read-only | Auto execute | Search synthetic policy index metadata |
| `save_draft` | L2 controlled write | Requires approval | Save a draft record after teacher approval |

Example registry setup:

```python
from app.services import EduFlowStore
from app.tools import build_mock_tool_registry

store = EduFlowStore()
store.initialize()
registry = build_mock_tool_registry(store)
```

Example read-only tool call:

```python
result = registry.execute(
    "get_class_profile",
    {"class_id": "kangaroo-room"},
)
```

Example controlled write with approval:

```python
result = registry.execute(
    "save_draft",
    {
        "draft_id": "draft-001",
        "draft_type": "activity_plan",
        "title": "Outdoor sensory walk",
        "content": "Synthetic draft content.",
    },
    approved=True,
)
```

## Repository Layout

```text
app/
  main.py          # FastAPI entry point and health endpoint
  config.py        # Runtime settings
  schemas/         # Pydantic request, response, and graph state models
  workflows/       # LangGraph workflows and nodes
  agents/          # Agent-specific orchestration modules
  tools/           # Tool definitions, registry, and executors
  services/        # SQLAlchemy-backed store and shared integrations
docs/              # Architecture notes and project documentation
tests/             # Unit and integration tests
data/synthetic/    # Synthetic demo data only
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies after `requirements.txt` is populated:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app.main:app --reload
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Run tests:

```bash
pytest
```
