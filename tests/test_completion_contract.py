from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
    resolve_required_controlled_tools,
)


class Empty(BaseModel):
    pass


def _controlled(name: str, aliases: tuple[str, ...]) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        category=ToolCategory.DRAFT,
        input_model=Empty,
        output_model=Empty,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        domain=ToolDomain.LOCAL,
        completion_aliases=aliases,
        handler=lambda _: ToolResult.ok(
            data={}, risk_level=RiskLevel.L2_CONTROLLED_WRITE
        ),
    )


def test_contract_is_derived_from_registered_controlled_tools_not_main_output():
    tools = [
        _controlled("save_record", ("保存教育记录",)),
        _controlled("publish_record", ("发布教育记录",)),
    ]

    assert resolve_required_controlled_tools("请保存教育记录", tools) == [
        "save_record"
    ]
    assert resolve_required_controlled_tools("不要保存教育记录", tools) == []


def test_future_controlled_operation_uses_the_same_generic_contract():
    tool = _controlled("send_to_family", ("发送给家庭",))

    assert resolve_required_controlled_tools("请发送给家庭", [tool]) == [
        "send_to_family"
    ]
