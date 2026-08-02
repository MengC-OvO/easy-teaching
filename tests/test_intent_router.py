from app.agents import INTENT_ROUTER_SYSTEM_PROMPT, IntentRouter
from app.schemas import Intent, IntentRouteResult
from app.services import ModelResponse


class StubStructuredProvider:
    def __init__(self, result: IntentRouteResult) -> None:
        self.result = result
        self.messages = None
        self.response_model = None
        self.temperature = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        self.temperature = temperature
        return ModelResponse(
            content=self.result.model_dump_json(),
            structured=self.result,
        )


class UnexpectedProviderCall:
    def generate_structured(self, **kwargs):
        raise AssertionError("Greeting-only routing should not call the model")


def test_intent_router_prompt_lists_supported_intents() -> None:
    assert "activity_planning" in INTENT_ROUTER_SYSTEM_PROMPT
    assert "learning_record" in INTENT_ROUTER_SYSTEM_PROMPT
    assert "policy_qa" in INTENT_ROUTER_SYSTEM_PROMPT
    assert "family_communication" in INTENT_ROUTER_SYSTEM_PROMPT
    assert "Return only valid JSON" in INTENT_ROUTER_SYSTEM_PROMPT


def test_intent_router_calls_provider_with_structured_schema() -> None:
    result = IntentRouteResult(
        intent=Intent.ACTIVITY_PLANNING,
        confidence=0.92,
        reason="The request asks for an activity plan.",
    )
    provider = StubStructuredProvider(result)

    routed = IntentRouter(provider).route("Plan a sensory activity for preschool children.")

    assert routed == result
    assert provider.response_model is IntentRouteResult
    assert provider.temperature == 0.0
    assert provider.messages[0].role.value == "system"
    assert provider.messages[1].content == "Plan a sensory activity for preschool children."


def test_intent_router_returns_clarification_result() -> None:
    result = IntentRouteResult(
        intent=Intent.UNKNOWN,
        confidence=0.4,
        needs_clarification=True,
        clarification_question="Do you want an activity plan or a family message draft?",
        reason="The request could map to more than one workflow.",
    )
    provider = StubStructuredProvider(result)

    routed = IntentRouter(provider).route("Can you help me write something for tomorrow?")

    assert routed.needs_clarification is True
    assert routed.intent is Intent.UNKNOWN


def test_intent_router_handles_greeting_without_guessing_a_workflow() -> None:
    routed = IntentRouter(UnexpectedProviderCall()).route("Hi!")

    assert routed.intent is Intent.UNKNOWN
    assert routed.needs_clarification is True
    assert "activity plan" in routed.clarification_question


def test_intent_router_does_not_intercept_greeting_with_a_real_request() -> None:
    result = IntentRouteResult(
        intent=Intent.POLICY_QA,
        confidence=0.94,
        reason="The message contains a policy question.",
    )
    provider = StubStructuredProvider(result)

    routed = IntentRouter(provider).route("Hi, what does the EYLF say about play?")

    assert routed.intent is Intent.POLICY_QA
    assert provider.messages is not None
