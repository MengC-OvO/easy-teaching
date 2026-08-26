from app.schemas import (
    CapabilityObservation,
    CapabilitySource,
    ObservationStatus,
)
from app.services import build_model_observation_view


def _observation(key: str, name: str, data):
    return CapabilityObservation(
        result_key=key,
        capability_name=name,
        source_kind=CapabilitySource.TOOL,
        status=ObservationStatus.COMPLETED,
        data=data,
    )


def test_model_view_preserves_rag_evidence_and_citation_but_drops_scores() -> None:
    original = _observation(
        "policy",
        "retrieve_knowledge",
        {
            "query": "agency",
            "strategy": "enhanced",
            "knowledge_scope": "eylf",
            "returned_count": 1,
            "search_queries": ["agency", "child agency"],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "content": "A" * 2_000,
                    "citation": {"source_id": "eylf-v2", "page": 12},
                    "bm25_score": 9.3,
                    "dense_distance": 0.12,
                    "final_rank": 1,
                    "metadata": {"large": "B" * 2_000},
                }
            ],
        },
    )

    view = build_model_observation_view({"policy": original})

    evidence = view["policy"]["data"]["evidence"][0]
    assert evidence["citation"] == {"source_id": "eylf-v2", "page": 12}
    assert evidence["content"].endswith("…[truncated]")
    assert "bm25_score" not in evidence
    assert "metadata" not in evidence
    assert len(original.data["evidence"][0]["content"]) == 2_000


def test_old_observations_become_summaries_while_recent_results_stay_detailed() -> None:
    observations = {
        f"item-{index}": _observation(
            f"item-{index}",
            "query_records",
            {"returned_count": index, "records": [{"title": f"Record {index}"}]},
        )
        for index in range(6)
    }

    view = build_model_observation_view(observations)

    assert "summary" in view["item-0"]
    assert "data" not in view["item-0"]
    assert view["item-5"]["data"]["records"][0]["title"] == "Record 5"


def test_record_view_keeps_only_fields_needed_for_reasoning() -> None:
    observation = _observation(
        "records",
        "query_records",
        {
            "returned_count": 1,
            "records": [
                {
                    "record_id": "record-1",
                    "title": "Garden story",
                    "analysis": "Observed learning",
                    "internal_audit_blob": "secret implementation detail",
                }
            ],
        },
    )

    view = build_model_observation_view({"records": observation})

    record = view["records"]["data"]["records"][0]
    assert record["record_id"] == "record-1"
    assert "internal_audit_blob" not in record


def test_drive_search_view_preserves_the_results_text_field() -> None:
    observation = _observation(
        "drive",
        "search_google_drive",
        {
            "results_text": "EasyTeaching evaluation export.docx",
            "provider": "google_drive",
        },
    )

    view = build_model_observation_view({"drive": observation})

    assert (
        view["drive"]["data"]["results_text"]
        == "EasyTeaching evaluation export.docx"
    )


def test_standard_rag_view_keeps_only_top_three_model_evidence_items() -> None:
    observation = _observation(
        "policy",
        "retrieve_knowledge",
        {
            "strategy": "standard",
            "returned_count": 5,
            "evidence": [
                {
                    "evidence_id": f"E{index}",
                    "content": "Evidence",
                    "citation": {"source_id": "eylf-v2"},
                    "final_rank": index,
                }
                for index in range(1, 6)
            ],
        },
    )

    view = build_model_observation_view({"policy": observation})

    assert len(view["policy"]["data"]["evidence"]) == 3
    assert observation.data["returned_count"] == 5


def test_draft_artifact_view_preserves_complete_bounded_content() -> None:
    content = "Complete draft paragraph. " * 120
    observation = _observation(
        "draft",
        "read_draft_artifact",
        {
            "source_request_id": "request-1",
            "title": "Nature sensory activity",
            "content": content,
            "content_chars": len(content),
            "created_at": "2026-08-25T09:00:00",
            "status": "unsaved",
        },
    )

    view = build_model_observation_view({"draft": observation})

    assert view["draft"]["data"]["content"] == content
    assert view["draft"]["data"]["content_chars"] == len(content)


def test_safety_view_returns_the_exact_checked_candidate_to_main() -> None:
    checked = "Complete activity\nKeep this exact closing line."
    observation = _observation(
        "safety",
        "check_activity_safety",
        {
            "status": "passed",
            "issues": [],
            "recovery_content": checked,
            "content_fingerprint": "a" * 64,
        },
    )

    view = build_model_observation_view({"safety": observation})

    assert view["safety"]["data"]["checked_activity_text"] == checked
    assert view["safety"]["data"]["content_fingerprint"] == "a" * 64
