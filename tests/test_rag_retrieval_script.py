from app.schemas import KnowledgeScope
from scripts.test_rag_retrieval import build_scope_filters, format_location, preview


def test_build_scope_filters_maps_teacher_facing_scopes_to_source_ids() -> None:
    assert build_scope_filters(KnowledgeScope.ALL).source_ids == []
    assert build_scope_filters(KnowledgeScope.EYLF).source_ids == ["eylf-v2"]
    assert build_scope_filters(KnowledgeScope.NQS).source_ids == ["nqs-guide-qa1"]
    assert build_scope_filters(KnowledgeScope.CENTRE_POLICY).source_ids == [
        "synthetic-centre-policies"
    ]


def test_format_location_and_preview_are_terminal_friendly() -> None:
    assert format_location("Learning through play", 21) == (
        "section=Learning through play, page=21"
    )
    assert preview("one  two\nthree", max_chars=20) == "one two three"
    assert preview("abcdefghijklmnopqrstuvwxyz", max_chars=10) == "abcdefg..."
