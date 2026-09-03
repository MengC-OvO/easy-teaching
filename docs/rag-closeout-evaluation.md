# RAG final closeout evaluation

Date: 2026-08-31

## Bottom line

The deterministic evidence threshold remains in place, but the fixed
false-premise-to-official-query rule map has been removed. Any production-path
metrics collected while that map was active are marked historical below and must
be rerun before they are presented as current results.

The resume benchmark is reproducible, but it is a 40-question fixed retrieval
benchmark, not the project-wide production result. The broader 280-question
production-path result must be reported separately.

## What changed

- Added a deterministic evidence gate between retrieval and answer generation.
  Weak evidence is removed; if nothing passes, the graph returns a fixed
  insufficient-evidence response before answer generation.
- Added focused-query extraction for long questions with retrieval-control
  preambles, so meta wording does not dominate reranking.
- Fixed multi-query fusion to retain the best Dense and BM25 evidence observed
  across rewrites instead of retaining the first query's scores.
- Removed the temporary fixed false-premise/official-concept query map. Questions
  now proceed through ordinary retrieval and the evidence threshold only.
- Recalibrated only the EYLF moderate Dense ceiling from 0.30 to 0.31. The lexical
  requirement remains BM25 >= 13; NQS and centre-policy thresholds are unchanged.
- Added Hit@K, Precision@K, MAP, confusion matrix, allow precision/recall/F1,
  specificity, balanced accuracy, false-allow/false-reject, answer/correction/
  abstention rates, citation checks, latency percentiles, and corpus-health checks.
- Added repeatable production retrieval, corpus audit, blind gate, robustness,
  and Gemini end-to-end evaluation scripts.

Current gate rules:

| Corpus/path | Pass rule |
|---|---|
| EYLF | Dense <= 0.22, or Dense <= 0.31 and BM25 >= 13 |
| NQS | Dense <= 0.22, or Dense <= 0.30 and BM25 >= 15 |
| Centre policy | Dense <= 0.30 and BM25 >= 22 |
| Deep | Corpus rule plus reranker >= -4.40 |

Dense distance is lower-is-better. BM25 and reranker scores are
higher-is-better. These thresholds are corpus/model-specific and require new
calibration if the corpus, chunking, embedding, or reranker changes.

## Evaluation coverage

| Layer | Cases | Mode executions | What it tests |
|---|---:|---:|---|
| Resume retrieval benchmark | 40 | 80 | Exact frozen gold chunks used by the resume claim |
| Broad production retrieval | 280 | 560 | EYLF, all NQS quality areas, NQF, centre policy, typo, short queries, distractors, Chinese |
| Clean gate calibration | 40 | 80 | Threshold fit; not used as headline evidence |
| Frozen blind gate validation | 84 | 168 | 54 answerable, 12 correctable, 18 unanswerable; no retuning after results |
| Gate robustness | 120 | 240 | Typo, telegraphic and distractor forms |
| Gemini end to end | 40 | 80 | Retrieval, gate, answer/correction/refusal, citations and quality judge |
| Corpus audit | 2,087 chunks | 1 audit | Vector/corpus consistency, metadata, duplicates and chunk sizes |
| Full regression | 358 tests | 358 | Entire project test suite |

These suites intentionally overlap at different layers. A retrieval question may
also have gate and answer tests because those detect different failures.

## 1. Retrieval results

### Fixed 40-question resume benchmark

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | Scope violations | Citation IDs valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard/Hybrid | 77.5% | 95.0% | 97.5% | 97.5% | 0.860 | 0.890 | 0% | 100% |
| Deep/Cross-encoder | 77.5% | **97.5%** | 97.5% | 100% | **0.869** | **0.902** | **0%** | **100%** |

This reproduces the resume numbers. “Citation IDs valid” means returned citation
metadata maps to indexed chunks; it does not by itself prove that the final prose
is factually correct.

### Broad 280-question production-path benchmark (historical; rerun required)

The following run used the now-removed fixed correction-query map. It remains useful
as an experiment record, but it is not a current production claim.

| Mode | R@1 | R@3 | R@5 | R@10 | MRR/MAP | nDCG@10 | Scope violations | Citation IDs valid | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | **77.9%** | **90.4%** | **94.3%** | 96.8% | **0.849** | **0.878** | 0% | 100% | 523 / 561 ms |
| Deep | 65.7% | 87.9% | 91.4% | **98.9%** | 0.778 | 0.830 | 0% | 100% | 2,624 / 5,260 ms |

Deep expands coverage by rank 10, but currently harms the important early ranks.
This is a reranking issue, not a claim that the correct evidence is absent.

By source scope:

| Scope | Mode | Cases | R@1 | R@3 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| EYLF | Standard | 110 | 82.7% | 98.2% | 100% | 0.905 |
| EYLF | Deep | 110 | 70.0% | 89.1% | 98.2% | 0.799 |
| NQS/NQF | Standard | 164 | 73.8% | 84.8% | 94.5% | 0.807 |
| NQS/NQF | Deep | 164 | 61.6% | 86.6% | 99.4% | 0.755 |
| Centre policy | Standard/Deep | 6 | 100% | 100% | 100% | 1.000 |

Robust query forms:

| Form (58 each) | Standard R@3 / R@10 | Deep R@3 / R@10 |
|---|---:|---:|
| Typo | 89.7% / 94.8% | 86.2% / 98.3% |
| Telegraphic | 87.9% / 96.6% | 91.4% / 98.3% |
| Distractor preamble | 89.7% / 96.6% | 87.9% / 100% |

The six Chinese cases are too small for a stable conclusion, but expose a real
Deep weakness: Standard R@3 was 83.3%, Deep R@3 was 33.3%.

Because almost every retrieval case has one exact gold chunk, P@10 is naturally
near 10% even on a perfect hit. R@K/Hit@K, MRR and nDCG are the meaningful ranking
metrics for this benchmark.

## 2. Evidence-gate and refusal results (historical; rerun required)

The following gate runs used production retrieval while the fixed correction-query
map was active. They must not be treated as current results after its removal.

### Clean calibration (40 cases)

Both modes allowed 100% of answerable and correctable cases and rejected 100% of
unanswerable cases. This validates the tuned examples only and is not treated as
the release headline.

### Frozen blind validation (84 cases)

| Mode | Answerable allowed | Correctable allowed | Unanswerable rejected | Allow precision | Allow recall/F1 | False allow | False reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 85.2% | 58.3% | **100%** | **100%** | 80.3% / 89.1% | **0%** | 19.7% |
| Deep | **92.6%** | **75.0%** | 94.4% | 98.3% | **89.4% / 93.7%** | 5.6% | **10.6%** |

Standard is conservative: no unsupported blind question was allowed, but it still
over-refuses some valid/correctable questions. Deep is more usable but falsely
allowed one fabricated programming-language requirement. Thresholds were not
retuned after seeing this blind result.

### Robustness (120 cases)

| Mode | Answerable allowed | Correctable allowed | Unanswerable rejected | F1 | False allow |
|---|---:|---:|---:|---:|---:|
| Standard | 91.7% | 71.4% | 100% | 90.9% | 0% |
| Deep | 95.0% | 76.2% | 100% | 93.2% | 0% |

These figures were produced before the fixed correction-query map was removed and
are retained as historical evidence only. The affected retrieval and gate suites
must be rerun before using them as current production claims.

## 3. Gemini end-to-end results (historical; rerun required)

The 40 public EYLF/NQS cases ran through production retrieval and the evidence
gate. Gemini generated an answer only when the gate allowed evidence; otherwise
the test used the fixed refusal. A separate Gemini call judged correctness,
groundedness, completeness and critical errors. Deterministic checks validated
answer type, required details, forbidden claims and citation IDs.

| Mode | Strict overall | Ordinary answers | Corrections | True refusals | Answerability accuracy | Citation-ID validity | Citation-policy pass | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 87.5% | 95.0% | 71.4% | **100%** | 97.5% | 100% | 100% | 3.19 / 4.03 s |
| Deep | 87.5% | 95.0% | 71.4% | **100%** | 97.5% | 100% | 97.5% | 5.25 / 7.30 s |

The model judge rated the generated content 5/5 for correctness, groundedness and
completeness with 0 critical errors. The stricter 87.5% result remains the headline:
four case families failed exact required-detail checks even though their prose was
semantically acceptable, and nonexistent NQS Element 8.2.4 was described as
unverifiable/abstained instead of explicitly labelled as a correction. This also
shows why a same-family model judge must not be the only metric.

## 4. Corpus/data quality

- Processed corpus and vector store both contain exactly 2,087 chunks.
- 3 sources: EYLF 134, NQS/NQF 1,949, synthetic centre policy 4.
- 0 duplicate chunk IDs, 0 empty chunks, 43 duplicate content hashes.
- 157 chunks are under 80 characters; one is over 4,000 characters.
- Metadata completeness is 99.968%; four missing URIs are the synthetic policy
  chunks.

The duplicate/very-short/oversized chunks are cleanup opportunities and likely
contributors to noisy NQS ranking, but they do not break index consistency.

## Release decision and remaining work

- Keep Standard as the default production path because it uses the simpler retrieval
  route. Current false-allow and false-reject rates are pending a post-removal rerun.
- Keep Deep available as an explicit broad-research path, but do not claim it is
  generally superior until cross-encoder ordering is fixed and reevaluated.
- Next retrieval work should target NQS broad/semantic questions, Chinese queries,
  and Deep rank fusion. Do not lower the gate globally to hide those problems.
- A production release should add human-labelled traffic samples and a judge from
  a different model/provider before claiming external validity.

## Resume wording

The original numbers may be kept only with the benchmark scope stated explicitly:

> Built tiered RAG with Hybrid retrieval and Query Rewrite + Cross-Encoder reranking;
> on a frozen 40-query exact-chunk benchmark, Deep achieved Recall@3 97.5%, MRR
> 0.869 and nDCG@10 0.902, with 0% scope violations and 100% citation-ID validity.

For a project-wide discussion, also disclose the 280-query production-path result:
Standard R@3 90.4% / MRR 0.849 / nDCG@10 0.878; Deep R@3 87.9% / MRR 0.778 /
nDCG@10 0.830.

## Verification and artifacts

Full project regression after removing the rule map: **357 passed, 1 third-party
deprecation warning, 0
failed**.

Raw reports are stored in the adjacent `eduflow-au-agent` evaluation workspace:

- `rag_final_production_retrieval_280.json`
- `rag_final_resume_standard_metrics.json`
- `rag_final_resume_deep_metrics.json`
- `rag_final_gate_clean_v2.json`
- `rag_final_gate_blind_validation.json`
- `rag_final_gate_robustness_v2.json`
- `rag_final_gemini_end_to_end_v2.json`
- `rag_closeout_corpus_audit.json`

## Limitations

- Expanded and blind questions were authored for this project, not supplied by an
  independent industry benchmark or external human panel.
- Robustness variants are systematic perturbations; real user traffic may differ.
- Centre-policy content was excluded from Gemini answer judging to avoid sending
  private/local evidence to an external provider.
- Answer and judge were separate calls but used the configured Gemini model family.
- Exact-gold-chunk retrieval can mark a semantically equivalent chunk wrong; future
  evaluation should add graded relevance with multiple acceptable chunks.
