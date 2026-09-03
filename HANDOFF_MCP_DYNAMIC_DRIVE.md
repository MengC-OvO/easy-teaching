# EasyTeaching MCP Dynamic Drive Handoff

## Goal

Replace the two model-facing Drive tools with one registered `drive_operation`
gateway:

1. `action=discover` lazily calls MCP `tools/list` and returns remote tool names,
   descriptions, schemas, and locally resolved risk metadata to Main.
2. On the next ReAct step, Main calls the same registered gateway with
   `action=execute`, `tool_name`, and `arguments`.
3. The gateway validates arguments against the discovered MCP JSON Schema and
   forwards the selected call through `MCPClient.call_tool`; remote tools do not
   need permanent local Registry entries.
4. Discovery/read-only operations auto-execute, controlled writes enter the
   existing approval flow, and destructive/public-sharing operations are denied.

## Changes already made

- `app/tools/mcp_adapter.py`
  - Added lazy `tools/list`, tool catalog caching (5 minutes), pagination, tool
    annotations, and base MCP risk classification.
- `app/tools/controlled_tools/google_drive.py`
  - Replaced separate search/upload builders with `build_google_drive_tool`.
  - Added `GoogleDriveMCPGateway` and two-stage discover/execute behavior.
  - Removes backend-owned `user_google_email` from model-facing remote schemas.
  - Keeps a safe `export_id` schema overlay for `create_drive_file`.
  - Keeps teacher/class scoping, managed export-root checks, SHA-256 validation,
    Base64 encoding, and no arbitrary local path support.
- `app/tools/definition.py` and `app/tools/registry.py`
  - Added argument-dependent permission and risk resolvers so discovery can be L0
    while a later write through the same registered gateway can be L2.
- `app/agents/main_react_agent.py`
  - Added the required two-step Drive instruction.
- `app/agents/main_react_executor.py` and `app/workflows/main_react_graph.py`
  - Decision validation and approval routing now use argument-resolved permission.
- `app/services/observation_view.py`
  - Preserves discovered Drive MCP schemas for the next model step.
- Registry builder, exports, availability, completion contract, docs, and tests
  were updated for the single `drive_operation` tool.

## Verification completed

- 37 focused registry/gateway tests passed.
- 99 broader MCP, Registry, Agent, graph, API, and observation-view tests passed.
- A live read-only smoke test started the installed `workspace-mcp`, called only
  `tools/list`, and discovered 9 Drive/core tools. It used the repository-local
  `.test-tmp` directory and did not open OAuth or call a Google API.
- 61 final focused MCP/Drive/Agent regression tests passed.
- The full project suite passed: 383 passed, 4 skipped.
- Import smoke checks passed.
- Warnings were pre-existing pytest `cache_dir` and LangChain pending-deprecation
  warnings.

## Final hardening completed

- Execute now requires a tool from the preceding discovered catalog and does not
  re-list between permission routing and invocation. This closes a catalog-change
  time-of-check/time-of-use gap and enforces the documented two-stage protocol.
- Remote `file_url` payloads are denied alongside paths, Base64 payloads, and the
  existing `fileUrl` spelling. The safe export overlay rejects extra fields.
- The two evaluation fixtures were migrated from the removed two-tool builder to
  the single dynamic gateway.
- Discovery intentionally returns all 9 Drive/core tools instead of hard-filtering
  by intent. The catalog is small, full schemas prevent capability false negatives,
  and each entry carries locally resolved risk and permission metadata.
- `scripts/test_google_drive_mcp.py --list-tools-only` now provides a repeatable
  catalog-only live check.

The repository still contains unrelated uncommitted Redis/MQ/RAG work that
predated this MCP change. Those changes were preserved.

## Important architecture wording

This project uses structured JSON decisions, not the model provider's native
function-calling `tools` field. Therefore discovered schemas are returned in the
`drive_operation` Observation and placed in the next Main prompt. Main then calls
the same locally registered gateway with the selected remote tool name. The remote
tool itself is not registered in the local `ToolRegistry`.
