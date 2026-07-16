from pathlib import Path


def test_readme_documents_core_product_scope() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Activity planning drafts" in readme
    assert "Learning record drafts" in readme
    assert "Policy question answering with citations" in readme
    assert "Family communication drafts" in readme


def test_readme_documents_safety_boundaries_and_risk_levels() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "must not" in readme
    assert "Diagnose children" in readme
    assert "medical advice" in readme
    assert "legal compliance conclusions" in readme
    assert "Send real messages to families" in readme
    assert "L0 read-only" in readme
    assert "L1 draft" in readme
    assert "L2 controlled write" in readme
    assert "L3 forbidden or handoff" in readme
    assert "Human approval is required" in readme
