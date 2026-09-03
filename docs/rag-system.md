# RAG system and retrieval evaluation

Updated: 2026-08-31

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
8. `retrieve_knowledge(mode=standard)` performs one-query Hybrid retrieval.
   `retrieve_knowledge(mode=deep)`
   rewrites the question into multiple queries, retrieves them separately, and
   applies a second RRF fusion followed by local Cross-encoder reranking.
   The final order blends the proven Hybrid/multi-query rank and Cross-encoder
   rank at 2:1 rather than trusting the generic reranker alone.
9. Both tools support hard `knowledge_scope` boundaries: `eylf`, `nqs`,
   `centre_policy`, or `all`. The scopes map to stored source IDs, so changing
   scope does not require rebuilding embeddings.
10. Teacher-uploaded centre documents use a separate approval-gated path. Each
    teacher/class scope has isolated SQLite FTS5 and Chroma indexes; a local
    SentenceTransformer creates 384-dimensional Dense vectors, and scoped search
    combines Dense 0.60 and BM25 0.40 with RRF. Uploaded private text is never sent
    to the external Gemini embedding provider.

## Deterministic evidence gate

`retrieve_knowledge` now filters the ranked candidate pool through calibrated
absolute relevance thresholds, then takes the requested Top-K from the passing
chunks before evidence reaches Main. This allows a lower-ranked passing chunk to
backfill a higher-ranked rejected chunk without ever padding the result with weak
evidence. Standard mode uses raw Dense
distance plus BM25; Deep mode requires the same base gate and a Cross-encoder
score of at least -4.40. RRF scores are deliberately excluded because they are
rank-only values. BM25 thresholds are source-aware because the EYLF, NQF guide,
and small centre-policy corpus have different score distributions. Tenant-local
uploaded indexes use their own conservative Dense ceiling plus Dense/BM25 rank
agreement because BM25 magnitude is not comparable across tiny private corpora.

If no chunk passes, the Tool returns `answerability=insufficient` with no
evidence. The graph then returns a fixed insufficient-evidence response without
another Main model call, so irrelevant Top-K neighbours cannot be converted into
an answer or citation. The gate is intentionally conservative: it protects
unsupported-answer cases, but some false-premise questions that could have been
corrected from indirect counter-evidence will instead be refused.

Thresholds were frozen using labelled answerable, correctable, and unanswerable
cases, then checked on frozen blind and robustness sets. EYLF currently passes at
Dense <= 0.22, or Dense <= 0.31 with BM25 >= 13. NQS passes at Dense <= 0.22, or
Dense <= 0.30 with BM25 >= 15. Centre policy requires Dense <= 0.30 and BM25 >=
22. Run `scripts/run_rag_gate_evals.py` to reproduce the production-path decision
test.

Long questions with an explicit retrieval-control preamble are reduced to their
factual question before retrieval. Deep multi-query fusion preserves the best
absolute Dense/BM25 evidence found by any rewrite. There is no special false-premise
prompt, framework-identifier rule, or correction-query map; questions proceed
through ordinary retrieval and the evidence threshold only.

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

Hybrid remains the default fast path. The 40-case benchmark shows a small
Cross-encoder improvement, but the broader 280-case production-path benchmark in
`docs/rag-closeout-evaluation.md` shows weaker early-rank Deep performance. Deep is
therefore reserved for explicit broad research while its rank fusion is improved.

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
40 and is accepted for the current release. It remains documented because hybrid
fusion improves average ranking but can still hurt an individual semantic result
when the lexical side is noisy.

## Evaluation boundary

The retrieval ranking benchmark is complete for the frozen evaluation corpus.
The separate final Agent suite covers end-to-end grounding, citations, Tool
selection and answer quality. Formal ablation of multi-query rewriting and a
larger independently authored human-faithfulness set remain future work; no
production certification is claimed.

## System review guide

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
7. Design trade-offs: why 768 dimensions, why 350/500/40 chunking, why local
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
