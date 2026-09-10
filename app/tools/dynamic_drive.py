"""Versioned Drive registrations reusing the existing controlled gateway."""
import hashlib
import json
import re
import asyncio

from pydantic import BaseModel, ConfigDict
from jsonschema.validators import validator_for

from app.tools.definition import ToolDefinition, ToolCategory, ToolDomain, ToolKind, ToolPermission
from app.tools.controlled_tools.google_drive import GoogleDriveMCPGateway, DriveOperationInput, DriveOperationOutput
from app.tools.definition import ToolResult
from app.schemas import RiskLevel


class LoadDriveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadDriveOutput(BaseModel):
    tool_names: list[str]
    message: str


def configure_lazy_drive(registry, store, *, client, user_google_email, timeout_seconds=45):
    """Register only a local loader. No MCP I/O during configuration."""
    if registry.get("load_drive_tools") is not None:
        if registry._dynamic_drive_account != user_google_email:
            raise PermissionError("A runtime catalog cannot be shared across Drive accounts")
        return
    registry._dynamic_drive_account = user_google_email
    lock = asyncio.Lock()
    cached = None

    async def ensure():
        nonlocal cached
        async with lock:
            if cached is None:
                cached = await asyncio.wait_for(register_dynamic_drive(registry, store,
                    client=client, user_google_email=user_google_email,
                    timeout_seconds=timeout_seconds), timeout=timeout_seconds)
            return cached

    registry.ensure_drive_tools = ensure

    async def load(data, context):
        catalog = await ensure()
        return ToolResult.ok(risk_level=RiskLevel.L0_READ_ONLY, data={
            "tool_names": [tool.name for tool in catalog],
            "message": "Drive tools loaded. Use their complete schemas on the next decision; loading did not execute a Drive operation.",
        })

    registry.register(ToolDefinition(name="load_drive_tools",
        description="Load Google Drive capabilities on demand. Call before searching, reading or uploading to Drive. This loads definitions only, not the requested operation.",
        category=ToolCategory.FILE, domain=ToolDomain.INTERNAL,
        input_model=LoadDriveInput, output_model=LoadDriveOutput,
        risk_level=RiskLevel.L0_READ_ONLY, permission=ToolPermission.AUTO_EXECUTE,
        permission_resolver=lambda raw: ToolPermission.AUTO_EXECUTE,
        completion_aliases=("upload to drive", "save to drive", "上传到drive", "保存到drive"),
        timeout_seconds=timeout_seconds, async_runtime_handler=load))
    legacy = registry.get("drive_operation")
    if legacy is not None:
        registry._tools[legacy.name] = legacy.model_copy(update={"model_visible": False})


async def tools_for_run(registry, observations):
    """Shared definitions, but exposure is granted by this run's loader result."""
    if registry.get("load_drive_tools") is None:
        return registry.list_tools()
    loaded = set()
    activated = False
    for item in observations.values():
        if item.capability_name == "load_drive_tools" and item.status.value == "completed":
            activated = True
            loaded.update(item.data.get("tool_names", []))
    if activated and any(registry.get(name) is None for name in loaded):
        # A resumed checkpoint may precede catalog reconstruction after restart.
        await registry.ensure_drive_tools()
    return [tool for tool in registry.list_tools()
        if (not tool.name.startswith("drive__") or tool.name in loaded)
        and not (activated and tool.name == "load_drive_tools")]


class DynamicArguments(BaseModel):
    model_config = ConfigDict(extra="allow")


def check_local_references(schema):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in {"$ref", "$dynamicRef"} and (not isinstance(value, str) or not value.startswith("#")):
                raise ValueError("Remote schema references are not allowed")
            check_local_references(value)
    elif isinstance(schema, list):
        for value in schema:
            check_local_references(value)


async def register_dynamic_drive(registry, store, *, client, user_google_email, timeout_seconds=45):
    """Build a new immutable catalog snapshot; publish registrations atomically.

    Called by the lazy loader. Publish versioned definitions without deleting or
    replacing definitions that active runs or frozen approvals may still use.
    """
    if getattr(registry, "_dynamic_drive_account", user_google_email) != user_google_email:
        raise PermissionError("A runtime catalog cannot be shared across Drive accounts")
    gateway = GoogleDriveMCPGateway(store, client=client, user_google_email=user_google_email)
    public = await gateway.discover()
    registrations = []
    for item in public:
        remote_name = item["name"]
        # Annotations alone are not a trusted local policy. Unknown tools stay
        # out of the dynamic catalog until explicitly reviewed.
        allowed_reads = {"search_drive_files", "get_drive_file_content", "get_drive_file_metadata", "list_drive_files"}
        if remote_name not in allowed_reads | {"create_drive_file"}:
            continue
        if item["permission"] == ToolPermission.FORBIDDEN.value:
            continue
        schema = item["input_schema"]
        check_local_references(schema)
        validator_for(schema).check_schema(schema)
        remote = gateway._catalog[remote_name]
        fingerprint = hashlib.sha256(json.dumps({
            "schema": remote.input_schema, "annotations": remote.annotations,
            "safe_schema": schema,
        }, sort_keys=True, default=str).encode()).hexdigest()
        name = "drive__" + re.sub(r"[^a-zA-Z0-9_]", "_", remote_name) + "__" + fingerprint[:12]

        def envelope(raw, remote_name=remote_name):
            return {"action": "execute", "tool_name": remote_name, "arguments": raw}

        async def execute(data, context, remote_name=remote_name, fingerprint=fingerprint):
            # Refresh remote definition before executing even a frozen approval.
            advertised = await client.list_tools(server_name=gateway.server_name, refresh=True)
            current = next((tool for tool in advertised if tool.name == remote_name), None)
            if current is None:
                raise ValueError("MCP tool is no longer available; create a new request")
            actual = hashlib.sha256(json.dumps({
                "schema": current.input_schema, "annotations": current.annotations,
                "safe_schema": gateway._safe_schema(current),
            }, sort_keys=True, default=str).encode()).hexdigest()
            if actual != fingerprint:
                raise ValueError("MCP definition changed; refresh catalog and request fresh approval")
            return await gateway.execute(DriveOperationInput(
                action="execute", tool_name=remote_name, arguments=data.model_dump(mode="json"),
            ), context)

        registrations.append(ToolDefinition(
            name=name, description=item["description"], category=ToolCategory.FILE,
            input_model=DynamicArguments, remote_input_schema=schema, output_model=DriveOperationOutput,
            risk_level=item["risk_level"], permission=(
                ToolPermission.REQUIRE_APPROVAL if remote_name == "create_drive_file" else ToolPermission.AUTO_EXECUTE
            ), kind=ToolKind.MCP, domain=ToolDomain.EXTERNAL,
            timeout_seconds=timeout_seconds, async_runtime_handler=execute,
            permission_resolver=lambda raw, wrap=envelope: gateway.permission_for(wrap(raw)),
            risk_resolver=lambda raw, wrap=envelope: gateway.risk_for(wrap(raw)),
            completion_aliases=("upload_export_to_google_drive", "upload to drive", "save to drive", "上传到drive", "保存到drive", "上传到google drive", "保存到google drive") if remote_name == "create_drive_file" else (),
        ))
    # Only touched during initialization; equivalent registrations are idempotent.
    replacement = dict(registry._tools)
    for tool in registrations:
        existing = replacement.get(tool.name)
        if existing is not None and existing.input_schema() != tool.input_schema():
            raise ValueError("Dynamic tool name collision")
        replacement[tool.name] = tool
    if "drive_operation" in replacement:
        replacement["drive_operation"] = replacement["drive_operation"].model_copy(update={"model_visible": False})
    registry._tools = replacement
    registry._dynamic_drive_account = user_google_email
    return registrations
