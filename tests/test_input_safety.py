import asyncio
from uuid import uuid4

from app.integrations.input_safety import InputSafetyRejected, prepare_user_input
from app.integrations.privacy_gateway_client import PrivacyGatewayUnavailableError
from safety_gateway.contracts import GatewayAction, InspectResponse, SafetySignals


def inspection(*, action="allow", redacted_text="Safe <PERSON_NAME_1>", mapping_id="opaque-mapping-id-123456789"):
    return InspectResponse(
        request_id=uuid4(),
        action=GatewayAction(action),
        reason_code=f"synthetic_{action}",
        signals=SafetySignals(
            injection_risk="normal" if action == "allow" else "suspicious",
            education_scope="in_scope",
            professional_risk="none",
        ),
        redacted_text=redacted_text if action == "allow" else None,
        mapping_id=mapping_id if action == "allow" else None,
        entity_counts={},
    )


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.inspected = []
        self.discarded = []

    async def inspect(self, request):
        self.inspected.append(request)
        if self.error:
            raise self.error
        return self.result

    async def discard(self, mapping_id):
        self.discarded.append(mapping_id)


def test_disabled_mode_never_calls_gateway() -> None:
    result = asyncio.run(
        prepare_user_input(mode="disabled", client=None, session_id="session", text="Synthetic input")
    )
    assert result.forwarded_text == "Synthetic input"
    assert result.mapping_id is None


def test_enforce_forwards_only_redacted_text_and_mapping_id() -> None:
    client = FakeClient(inspection())
    result = asyncio.run(
        prepare_user_input(mode="enforce", client=client, session_id=str(uuid4()), text="Synthetic raw")
    )
    assert result.forwarded_text == "Safe <PERSON_NAME_1>"
    assert result.mapping_id == "opaque-mapping-id-123456789"
    assert client.inspected[0].text == "Synthetic raw"


def test_enforce_rejects_non_allow_decision() -> None:
    client = FakeClient(inspection(action="block"))
    try:
        asyncio.run(
            prepare_user_input(mode="enforce", client=client, session_id="session", text="Synthetic attack")
        )
    except InputSafetyRejected as error:
        assert error.inspection.action is GatewayAction.BLOCK
    else:
        raise AssertionError("block decision must not be forwarded")


def test_shadow_discards_mapping_and_forwards_original() -> None:
    client = FakeClient(inspection())
    result = asyncio.run(
        prepare_user_input(mode="shadow", client=client, session_id="session", text="Synthetic original")
    )
    assert result.forwarded_text == "Synthetic original"
    assert result.mapping_id is None
    assert client.discarded == ["opaque-mapping-id-123456789"]


def test_shadow_tolerates_unavailable_gateway_but_enforce_does_not() -> None:
    error = PrivacyGatewayUnavailableError("synthetic unavailable")
    shadow = asyncio.run(
        prepare_user_input(
            mode="shadow",
            client=FakeClient(error=error),
            session_id="session",
            text="Synthetic original",
        )
    )
    assert shadow.forwarded_text == "Synthetic original"
    try:
        asyncio.run(
            prepare_user_input(
                mode="enforce",
                client=FakeClient(error=error),
                session_id="session",
                text="Synthetic original",
            )
        )
    except PrivacyGatewayUnavailableError:
        pass
    else:
        raise AssertionError("enforce must fail closed")
