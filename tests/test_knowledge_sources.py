import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "knowledge" / "sources.json"


def load_sources():
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def test_knowledge_source_manifest_exists_and_has_required_sources() -> None:
    sources = load_sources()

    assert {source["source_id"] for source in sources} == {
        "eylf-v2",
        "nqs-guide-qa1",
        "synthetic-centre-policies",
    }


def test_knowledge_source_files_exist() -> None:
    for source in load_sources():
        path = PROJECT_ROOT / source["path"]

        assert path.exists(), source["path"]
        assert path.stat().st_size > 0


def test_official_pdf_sources_are_pdf_files() -> None:
    official_pdfs = [
        source
        for source in load_sources()
        if source["source_type"] == "official" and source["format"] == "pdf"
    ]

    assert official_pdfs
    for source in official_pdfs:
        path = PROJECT_ROOT / source["path"]

        assert path.read_bytes().startswith(b"%PDF")


def test_synthetic_policy_source_is_clearly_marked() -> None:
    source = next(
        source
        for source in load_sources()
        if source["source_id"] == "synthetic-centre-policies"
    )
    content = (PROJECT_ROOT / source["path"]).read_text(encoding="utf-8")

    assert source["source_type"] == "synthetic"
    assert "synthetic demo materials" in content
    assert "not real centre policies" in content
    assert "must not send messages directly" in content
