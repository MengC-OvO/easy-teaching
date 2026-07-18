from app.services import EduFlowStore


def test_store_initializes_schema_and_seed_data(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")

    store.initialize()

    profile = store.get_class_profile("kangaroo-room")
    policies = store.search_policy_index("program")

    assert profile is not None
    assert profile["name"] == "Kangaroo Room"
    assert profile["age_group"] == "3-5"
    assert "outdoor play" in profile["interests"]
    assert policies[0]["policy_id"] == "nqs-qa1-program"


def test_store_saves_draft(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")
    store.initialize()

    saved = store.save_draft(
        draft_id="draft-001",
        draft_type="activity_plan",
        title="Outdoor sensory walk",
        content="Synthetic draft content.",
    )

    assert saved == {
        "draft_id": "draft-001",
        "draft_type": "activity_plan",
        "title": "Outdoor sensory walk",
        "status": "draft",
    }


def test_store_save_draft_is_idempotent_with_key(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")
    store.initialize()

    first = store.save_draft(
        draft_id="draft-001",
        draft_type="activity_plan",
        title="Outdoor sensory walk",
        content="Synthetic draft content.",
        idempotency_key="request-001:save-draft",
    )
    second = store.save_draft(
        draft_id="draft-002",
        draft_type="activity_plan",
        title="Should not replace original",
        content="Different synthetic content.",
        idempotency_key="request-001:save-draft",
    )

    assert second == first
    assert second["draft_id"] == "draft-001"
    assert second["title"] == "Outdoor sensory walk"


def test_store_initialization_is_idempotent(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")

    store.initialize()
    store.initialize()

    assert store.count_class_profiles() == 1
