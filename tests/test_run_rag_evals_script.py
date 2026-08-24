import json
from collections import Counter
from pathlib import Path

from app.config import settings
from app.services import KnowledgeIngestionService
from scripts.run_rag_evals import CachedQueryEmbeddingProvider, load_cases


class StubEmbeddingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY"):
        self.calls += 1
        return [0.25] * settings.embedding_dimension


class FailIfCalledEmbeddingProvider:
    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY"):
        raise AssertionError("cached query should not call the API provider")


def test_query_embedding_cache_persists_and_reuses_vectors(tmp_path) -> None:
    cache_path = tmp_path / "query_vectors.json"
    first = CachedQueryEmbeddingProvider(cache_path)
    stub = StubEmbeddingProvider()
    first.provider = stub

    vector = first.embed_text("play based learning")

    assert stub.calls == 1
    assert first.api_calls == 1
    assert len(first.api_latencies_ms) == 1
    assert len(vector) == settings.embedding_dimension
    assert json.loads(cache_path.read_text(encoding="utf-8"))["vectors"]

    second = CachedQueryEmbeddingProvider(cache_path)
    second.provider = FailIfCalledEmbeddingProvider()

    assert second.embed_text("play based learning") == vector
    assert second.cache_hits == 1
    assert second.api_calls == 0


def test_seed_cases_have_separate_development_and_test_splits() -> None:
    cases_path = Path("data/evals/rag_retrieval_cases.json")
    all_cases = load_cases(cases_path, "all")
    dev_cases = load_cases(cases_path, "dev")
    test_cases = load_cases(cases_path, "test")

    assert len(all_cases) == 12
    assert len(dev_cases) == 8
    assert len(test_cases) == 4
    assert {case.case_id for case in dev_cases}.isdisjoint(
        case.case_id for case in test_cases
    )


def test_final_cases_reference_real_chunks_in_the_expected_scope() -> None:
    cases = load_cases(Path("data/evals/rag_final_cases.json"), "test")
    ingestion = KnowledgeIngestionService(project_root=Path.cwd())
    chunks = ingestion.read_chunks_jsonl(Path("data/knowledge/processed/chunks.jsonl"))
    catalog = {chunk.chunk_id: chunk for chunk in chunks}

    assert len(cases) == 40
    assert Counter(case.scope.value for case in cases) == {
        "eylf": 18,
        "nqs": 18,
        "centre_policy": 4,
    }
    for case in cases:
        for gold in case.relevant_evidence:
            assert gold.chunk_id in catalog
            chunk = catalog[gold.chunk_id]
            assert chunk.document.source_id == gold.source_id
            if gold.page is not None:
                assert chunk.page == gold.page
            if gold.section_contains is not None:
                assert gold.section_contains.lower() in (chunk.section or "").lower()
