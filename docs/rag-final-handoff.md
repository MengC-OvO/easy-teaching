# RAG final handoff and interview review

Updated: 2026-08-23

## Current decision

Freeze the current layout-aware chunking configuration and use Hybrid retrieval
as the default production baseline. Do not spend more time on chunk-size
ablation unless a larger evaluation set later exposes a recall problem.

## Implemented pipeline

1. Source files are declared in `data/knowledge/sources.json`:
   EYLF V2.0, the NQF/NQS guide, and synthetic centre policies.
2. `KnowledgeIngestionService` parses PDFs with PyMuPDF4LLM without OCR, keeps
   headings and page numbers, normalises text, and removes duplicate chunk IDs.
3. Chunking uses target 350 approximate tokens, maximum 500, and overlap 40.
   The frozen processed corpus contains 2,087 unique chunks.
4. Every chunk carries stable `chunk_id`, content hash, source ID/type, document
   title, version, URI, section, page, parser, chunk position, and approximate
   token count.
5. Gemini `gemini-embedding-001` produces 768-dimensional document and query
   vectors. Chroma persists the HNSW cosine vector index under `data/chroma`.
6. SQLite FTS5 persists the BM25 inverted index under
   `data/knowledge/index/knowledge_fts.sqlite3`.
7. Hybrid retrieval obtains Dense and BM25 candidates, deduplicates them, and
   combines their ranks with weighted reciprocal-rank fusion (Dense 0.60,
   BM25 0.40, RRF k=60).
8. `search_knowledge` performs one-query Hybrid retrieval. `research_knowledge`
   rewrites the question into multiple queries, retrieves them separately, and
   applies a second RRF fusion followed by local Cross-encoder reranking.
   The final order blends the proven Hybrid/multi-query rank and Cross-encoder
   rank at 2:1 rather than trusting the generic reranker alone.
9. Both tools support hard `knowledge_scope` boundaries: `eylf`, `nqs`,
   `centre_policy`, or `all`. The scopes map to stored source IDs, so changing
   scope does not require rebuilding embeddings.

## Final retrieval evaluation

The frozen final set is `data/evals/rag_final_cases.json`:

- 40 test-only questions
- 18 EYLF, 18 NQS, and 4 synthetic centre-policy questions
- one manually selected exact gold `chunk_id` per question
- all gold chunks validated against the 2,087-chunk corpus

Final results:

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 | Scope error | Citation correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.625 | 0.825 | 0.925 | 0.975 | 0.740 | 0.797 | 0.000 | 1.000 |
| Dense | 0.675 | 0.950 | 0.975 | 1.000 | 0.810 | 0.858 | 0.000 | 1.000 |
| Hybrid | **0.775** | **0.950** | **0.975** | 0.975 | **0.860** | **0.890** | **0.000** | **1.000** |
| Hybrid + Cross-encoder | **0.775** | **0.975** | **0.975** | **1.000** | **0.869** | **0.902** | **0.000** | **1.000** |

Hybrid remains the default fast path. Cross-encoder reranking improves the
deeper ranking metrics without reducing first-result accuracy, but its CPU
latency is reserved for the explicit enhanced research tool.

Measured latency:

- Query embedding API: P50 about 542 ms, P95 about 575 ms.
- Hybrid local retrieval after query embedding: P50 16.5 ms, P95 27.2 ms.
- Hybrid + local Cross-encoder: P50 1,158.8 ms, P95 1,435.7 ms.
- Expected online Hybrid retrieval is therefore roughly 0.56-0.61 seconds.

The final run used 40 query-embedding API calls. Query vectors are cached under
ignored `data/local/`; rerunning the same suite should require zero API calls.

## Known boundary case

`final-nqs-02-standard-program` asks for the central requirement of NQS
Standard 1.1. Dense ranks the exact gold chunk third, but Hybrid misses it in
the top 10 because broad BM25 matches to NQS headings, contents, and regulatory
sections dominate the fused candidate list. This is one known failure out of
40 and is accepted for the current milestone. Do not hide it in an interview;
explain it as evidence that hybrid fusion improves average ranking but can hurt
one semantic result when the lexical side is noisy.

## Not yet claimed as complete

- End-to-end answer faithfulness and answer quality evaluation
- Full Agent tool-choice and orchestration evaluation
- Formal evaluation of multi-query rewriting in `research_knowledge`
- End-to-end evaluation of multi-query rewrite + retrieval + reranking as one
  enhanced tool path (the retrieval-only Cross-encoder stage is evaluated)

## Next interview review session

Review the project from start to finish in this order:

1. RAG purpose: parametric knowledge versus retrieved external evidence.
2. Ingestion: layout-aware parsing, normalisation, chunking, overlap, dedupe,
   stable IDs, metadata, and citations.
3. Index construction: Embedding/Chroma Dense index versus SQLite FTS5 BM25
   inverted index, including why BM25 needs no embedding.
4. Online query path: source-scope filter, query embedding, Dense and BM25
   candidate retrieval, weighted RRF, deduplication, ranking, and evidence.
5. Tool layer: simple versus enhanced retrieval and how the Agent calls them.
6. Evaluation: exact gold chunk labels, Recall@K, MRR, nDCG, scope violations,
   citation correctness, caching, P50/P95, and interpreting the final table.
7. Interview trade-offs: why 768 dimensions, why 350/500/40 chunking, why local
   Chroma and SQLite, why Hybrid is default, why Cross-encoder is enhanced, and
   how the system would scale beyond a single machine.
8. Practise a two-minute architecture explanation, debugging questions, and
   follow-up design questions about production scaling and evaluation.

## Commands

Standalone evidence inspection:

```powershell
.\.venv\Scripts\python.exe scripts\test_rag_retrieval.py `
  "What does the EYLF say about play-based learning?" `
  --scope eylf --mode hybrid --top-k 5
```

Final retrieval metrics:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evals.py `
  --cases data/evals/rag_final_cases.json `
  --split test --modes bm25 dense hybrid `
  --report reports/rag_final_report.json
```

Final enhanced reranking metrics (cached queries use no Gemini quota):

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evals.py `
  --cases data/evals/rag_final_cases.json `
  --split test --modes hybrid --reranker cross_encoder `
  --report reports/rag_final_rerank_report.json
```
