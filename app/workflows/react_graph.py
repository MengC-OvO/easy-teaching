from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Union

from langgraph.graph import END, StateGraph

from app.agents import ReActAgent, ReActToolExecutor
from app.schemas import ReActState, StopReason
from app.services import EduFlowStore, ModelProviderError
from app.tools import ToolDefinition, ToolRegistry, build_default_tool_registry


ReActStateInput = Union[ReActState, Mapping[str, Any]]


class ReActAgentProtocol(Protocol):
    def decide(self, state: ReActState, available_tools: List[ToolDefinition]):
        ...


def _state_from_input(state: ReActStateInput) -> ReActState:
    if isinstance(state, ReActState):
        return state
    return ReActState.model_validate(state)


def build_react_agent_node(
    agent: ReActAgentProtocol,
    registry: ToolRegistry,
    executor: ReActToolExecutor,
):
    def react_agent_node(state: ReActStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        if current_state.should_stop:
            return {}
        if not current_state.has_steps_remaining:
            return {
                "stop_reason": StopReason.MAX_STEPS_REACHED,
                "current_step": current_state.current_step,
            }

        try:
            available_tools = registry.list_tools(
                allowed_tool_names=executor.effective_allowed_tool_names(
                    current_state
                )
            )
            decision = agent.decide(current_state, available_tools)
        except ModelProviderError:
            return {
                "stop_reason": StopReason.MODEL_ERROR,
                "current_step": current_state.current_step,
            }

        return {"decision": decision}

    return react_agent_node


def build_react_tool_executor_node(
    executor: ReActToolExecutor,
    *,
    approved: bool = False,
):
    def react_tool_executor_node(state: ReActStateInput) -> Dict[str, object]:
        current_state = _state_from_input(state)
        return executor.execute(current_state, approved=approved)

    return react_tool_executor_node


def mark_max_steps_reached(state: ReActStateInput) -> Dict[str, object]:
    current_state = _state_from_input(state)
    return {
        "stop_reason": StopReason.MAX_STEPS_REACHED,
        "current_step": current_state.current_step,
    }


def route_after_agent(state: ReActStateInput) -> str:
    current_state = _state_from_input(state)
    if current_state.should_stop:
        return "end"
    return "tool_executor"


def route_after_tool_executor(state: ReActStateInput) -> str:
    current_state = _state_from_input(state)
    if current_state.should_stop:
        return "end"
    if not current_state.has_steps_remaining:
        return "max_steps_stop"
    return "agent"


def _default_registry() -> ToolRegistry:
    store = EduFlowStore()
    store.initialize()
    return build_default_tool_registry(store)


def build_react_graph(
    *,
    agent: Optional[ReActAgentProtocol] = None,
    registry: Optional[ToolRegistry] = None,
    allowed_tool_names: Optional[Iterable[str]] = None,
    approved: bool = False,
    required_skill_name: Optional[str] = None,
):
    resolved_registry = registry or _default_registry()
    resolved_agent = agent or ReActAgent()
    allowed_tool_name_set = _normalize_allowed_tool_names(allowed_tool_names)
    executor = ReActToolExecutor(
        resolved_registry,
        allowed_tool_names=allowed_tool_name_set,
        required_skill_name=required_skill_name,
    )

    graph = StateGraph(ReActState)
    graph.add_node(
        "agent",
        build_react_agent_node(resolved_agent, resolved_registry, executor),
    )
    graph.add_node(
        "tool_executor",
        build_react_tool_executor_node(executor, approved=approved),
    )
    graph.add_node("max_steps_stop", mark_max_steps_reached)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "tool_executor",
        route_after_tool_executor,
        {
            "agent": "agent",
            "max_steps_stop": "max_steps_stop",
            "end": END,
        },
    )
    graph.add_edge("max_steps_stop", END)
    return graph.compile()


def _normalize_allowed_tool_names(
    allowed_tool_names: Optional[Iterable[str]],
) -> Optional[Set[str]]:
    if allowed_tool_names is None:
        return None
    return set(allowed_tool_names)
