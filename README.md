# EduFlow AU Agent

EduFlow AU Agent is a learning project for building a teacher workflow agent
for Australian early childhood education scenarios. The current codebase
focuses on a runnable, testable agent backbone: intent routing, controlled
tools, ReAct execution, document ingestion, embeddings, and a local vector
index for RAG.

All project data should be synthetic, public, or thoroughly de-identified.

## Product Scope

The agent supports teacher-assistant workflows for:

- Activity planning drafts
- Learning record drafts
- Policy question answering with citations
- Family communication drafts

The agent may draft, search, and reason, but code-level validation controls
which tools can run, which writes require approval, and which actions are
forbidden.

## Architecture

```mermaid
flowchart TD
    User[Teacher request] --> API[FastAPI app]
    API --> MainGraph[LangGraph main graph]
    Checkpoint[(SQLite checkpointer)] -. thread state .-> MainGraph
    MainGraph --> GraphState[GraphState]
    ProfileMemory[Teacher profile memory] --> ContextManager[ContextManager]
    GraphState --> ContextManager
    ContextManager --> Router[IntentRouter]
    GraphState --> Router[IntentRouter]

    Router -->|activity_planning| PlanningReAct[Planning ReAct workflow]
    Router -->|learning_record| DocumentationSkeleton[Documentation specialist subgraph]
    Router -->|policy_qa| PolicyRAG[Policy RAG workflow]
    Router -->|family_communication| Family[Family specialist skeleton]
    Router -->|unknown or clarification| Clarification[Clarification response]

    PlanningReAct --> ReActAgent[ReActAgent]
    ReActAgent --> ModelProvider[ChatCompletionsModelProvider]
    ReActAgent --> ReActExecutor[ReActToolExecutor]
    ReActExecutor --> ToolRegistry[ToolRegistry]
    ToolRegistry --> ClassTool[get_class_profile]
    ToolRegistry --> EvidenceTool[retrieve_risk_guidance]
    ToolRegistry --> SafetyTool[check_activity_safety]
    ToolRegistry --> EylfTool[align_to_eylf_outcomes]
    ToolRegistry --> DraftTool[save_draft]
    ToolRegistry --> RecallTool[recall_long_term_memory]
    ClassTool --> Store[EduFlowStore SQLite]
    DraftTool --> Store
    RecallTool --> Store
    EvidenceTool --> PlanningRetrieval[KnowledgeRetriever Hybrid]
    EylfTool --> PlanningRetrieval
    SafetyTool --> SafetyRules[Deterministic safety rules]

    PolicyRAG --> PolicyService[PolicyRAGService]
    PolicyService --> Retrieval[KnowledgeRetriever]
    Retrieval --> Embeddings[GeminiEmbeddingProvider]
    Retrieval --> BM25[BM25KnowledgeIndex]
    Retrieval --> Reranker[Lexical or CrossEncoderReranker]
    Retrieval --> VectorStore[ChromaVectorStore]
    VectorStore --> Chroma[(Chroma collection)]
    PolicyService --> AnswerModel[ChatCompletionsModelProvider]
    PlanningReAct --> ContextUpdate[context_update]
    PolicyRAG --> ContextUpdate
    DocumentationSkeleton --> ContextUpdate
    Family --> ContextUpdate
    Clarification --> ContextUpdate
    ContextUpdate --> MemoryUpdate[long_memory_update]
    MemoryUpdate --> Store
```

## LangGraph Flow

```mermaid
flowchart TD
    Start([START]) --> Initialize[initialize]
    Initialize --> IntentNode[intent_router]
    IntentNode --> Route{route_by_intent}

    Route -->|activity_planning| PlanningSubgraph
    Route -->|policy_qa| PolicySubgraph
    Route -->|learning_record| Documentation[documentation specialist skeleton]
    Route -->|family_communication| Family[family specialist skeleton]
    Route -->|unknown / needs clarification| Clarification[clarification_placeholder]
    Route -->|router failed| ContextUpdate[context_update]

    subgraph PlanningSubgraph[Activity Planning ReAct Subgraph]
        PAgent[agent]
        PExecutor[tool_executor]
        PMax[max_steps_stop]
        PAgent -->|call tool| PExecutor
        PExecutor -->|continue| PAgent
        PExecutor -->|max steps| PMax
        PAgent -->|final / stop| PEnd([subgraph END])
        PExecutor -->|approval / error / complete| PEnd
        PMax --> PEnd
    end

    subgraph PolicySubgraph[Policy RAG Subgraph]
        PolicyNode[policy_rag]
        PolicyNode --> Retrieve[retrieve evidence]
        Retrieve --> Gate{evidence gate}
        Gate -->|empty retrieval| AskClarify[needs clarification]
        Gate -->|version conflict| Conflict[evidence conflict]
        Gate -->|grounded evidence| Generate[generate answer]
        Generate --> PolicyEnd([subgraph END])
        AskClarify --> PolicyEnd
        Conflict --> PolicyEnd
    end

    PlanningSubgraph --> ContextUpdate
    PolicySubgraph --> ContextUpdate
    Documentation --> ContextUpdate
    Family --> ContextUpdate
    Clarification --> ContextUpdate
    ContextUpdate --> MemoryUpdate[long_memory_update]
    MemoryUpdate --> End([END])
```

## Knowledge Pipeline

```mermaid
flowchart LR
    Sources[Official PDFs + synthetic markdown]
    Sources --> Ingestion[KnowledgeIngestionService]
    Ingestion --> ParsedBlocks[ParsedTextBlock]
    ParsedBlocks --> Chunks[KnowledgeChunk JSONL]
    Chunks --> Embedding[Gemini embedding]
    Embedding --> Vectors[768-d vectors]
    Chunks --> VectorStore[ChromaVectorStore]
    Vectors --> VectorStore
    VectorStore --> Collection[(eduflow_knowledge)]

    Query[Teacher question] --> QueryEmbedding[Query embedding]
    QueryEmbedding --> Collection
    Collection --> Retrieved[RetrievedKnowledgeChunk + citation]
```

Current vector index settings:

```text
index_method=hnsw
distance_metric=cosine
embedding_model_name=gemini-embedding-001
embedding_dimension=768
collection_name=eduflow_knowledge
```

Gemini creates embeddings. Chroma stores those vectors and uses HNSW with
cosine distance to retrieve semantically similar chunks.

## Key Modules

```text
app/main.py
  FastAPI entry point and /health endpoint.

app/config.py
  Runtime settings loaded from .env.

app/schemas/
  Pydantic models for graph state, intent routing, ReAct decisions, and
  knowledge chunks/citations.

app/workflows/
  LangGraph workflows. The main graph routes requests by intent and connects
  activity planning to the ReAct workflow.

app/agents/
  IntentRouter, ReActAgent, and ReActToolExecutor orchestration logic.

app/tools/
  ToolDefinition, ToolResult, ToolRegistry, and controlled tool modules.
  controlled_tools/get_class_profile.py reads class context.
  controlled_tools/retrieve_risk_guidance.py retrieves risk, safety, and regulatory guidance.
  controlled_tools/check_activity_safety.py checks activity safety risks.
  controlled_tools/align_to_eylf_outcomes.py aligns activities to EYLF outcomes.
  controlled_tools/save_draft.py saves approved drafts.
  controlled_tools/recall_long_term_memory.py reads task-specific durable memory
  within the active teacher/class scope.

app/services/
  Model provider, embedding provider, SQLAlchemy store, knowledge ingestion,
  and Chroma vector store.

data/knowledge/
  Source manifest, raw knowledge documents, and processed chunks JSONL.

scripts/
  Local smoke tests and utility scripts for ingestion, model calls, and vector
  index building.

tests/
  Unit and integration tests for the agent backbone.
```

## Week 2 State and Memory Delivery

### Day 5: Checkpoint, short-term context, and compaction

- `build_sqlite_checkpointer()` creates a LangGraph SQLite checkpointer, and
  `checkpoint_config(thread_id)` supplies the stable thread key at invocation.
- `ThreadContext` retains bounded recent turns, bounded tool traces, and a
  structured conversation summary. When the recent-turn or token budget is
  exceeded, `LLMContextSummarizer` compresses the older context rather than
  passing an unbounded transcript to the next model call.
- The checkpointer persists graph execution state; the compact context is the
  smaller prompt projection of that state. They solve different problems.

### Day 6: Long-term memory and recall

- A completed graph request calls the structured memory writer, which returns
  `noop`, `insert`, `update`, or `delete` operations.
- Stable teacher preferences can become profile memory and are automatically
  loaded by the main graph. Task-specific history remains recall-only and is
  available through the ReAct tool.
- Long-term records are scoped to a teacher or class and rechecked by the
  SQLite store before mutation or recall.

## Memory Design

EduFlow uses one SQLite long-term-memory store with two retrieval modes:

- **Profile memory** is limited to active, teacher-scoped preferences such as
  language and stable output format. The main graph loads up to four records,
  ordered by importance and recency, before routing and drafting.
- **Recall-only memory** holds task-specific durable history. The Planning ReAct
  workflow can call the L0 `recall_long_term_memory` tool when a query needs
  historical detail. The tool receives teacher/class ownership from graph state,
  never from model-generated arguments.

After each completed main-graph request, a structured LLM memory decision may
return `noop`, `insert`, `update`, or `delete`. The SQLite store rechecks the
active teacher/class scope before applying a mutation. Short-term thread context
and its compact summary remain separate from cross-session long-term memory.

## Safety Boundaries

EduFlow AU is a teacher assistant. It must not:

- Diagnose children
- Provide medical advice
- Provide legal compliance conclusions
- Send real messages to families
- Use raw real child or family private information

### Risk Levels

| Level | Typical action | System behavior |
| --- | --- | --- |
| L0 read-only | Search policy text, read synthetic class configuration | Execute automatically and record sources |
| L1 draft | Generate activity plans, learning records, or family message drafts | Execute automatically, clearly marked as Draft |
| L2 controlled write | Save, overwrite, or export records | Show the change and require teacher confirmation |
| L3 forbidden or handoff | Real sending, diagnosis, medical/legal judgment, raw PII | Refuse or hand off with a clear boundary explanation |

Human approval is required before any controlled write or real-world side
effect. Approval is treated as a scoped authorization boundary, not as a casual
review step after unrestricted model behavior.

## Tool System

Tools are not plain functions exposed directly to the model. Each tool is
described by a `ToolDefinition` with:

- `name`
- `description`
- `category`
- Pydantic `input_model`
- Pydantic `output_model`
- `risk_level`
- `permission`
- `handler`

The `ToolRegistry` is the code-level boundary between model intent and tool
execution. It handles duplicate tool names, tool lookup, Pydantic validation,
approval checks, forbidden-tool blocking, handler exception wrapping, and
structured `ToolResult` output.

Current controlled tools:

| Tool | Risk | Permission | Purpose |
| --- | --- | --- | --- |
| `get_class_profile` | L0 read-only | Auto execute | Read synthetic class profile data |
| `retrieve_risk_guidance` | L0 read-only | Auto execute | Retrieve local risk, safety, and regulatory guidance |
| `check_activity_safety` | L0 read-only | Auto execute | Check activity drafts for common safety risks |
| `align_to_eylf_outcomes` | L0 read-only | Auto execute | Map activity drafts to likely EYLF outcomes with evidence |
| `save_draft` | L2 controlled write | Requires approval | Save a draft record with an idempotency key |
| `recall_long_term_memory` | L0 read-only | Auto execute | Search task-specific memory for the current teacher/class only |

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy local environment variables:

```bash
cp .env.example .env
```

The real API key belongs only in `.env`, which is ignored by Git.

Required model and embedding settings:

```text
MODEL_BASE_URL
MODEL_CHAT_COMPLETIONS_PATH
MODEL_API_KEY
MODEL_NAME
MODEL_TIMEOUT_SECONDS

EMBEDDING_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_MODEL_NAME
EMBEDDING_DIMENSION
EMBEDDING_TIMEOUT_SECONDS

CHROMA_PATH
CHROMA_COLLECTION_NAME
```

## Run Commands

Run tests:

```bash
python -m pytest
```

Run the API locally:

```bash
uvicorn app.main:app --reload
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Run model smoke tests:

```bash
python scripts/model_smoke_test.py
python scripts/intent_router_smoke_test.py
python scripts/gemini_week1_trace.py
```

Ingest knowledge sources into chunks:

```bash
python scripts/ingest_knowledge.py --output data/knowledge/processed/chunks.jsonl
```

Build a small vector index smoke test:

```bash
python scripts/build_vector_index.py --reset --limit 5 --batch-size 5
```

Continue building the full vector index without deleting existing chunks:

```bash
python scripts/build_vector_index.py --batch-size 4 --max-retries 10 --retry-delay-seconds 15 --batch-delay-seconds 3
```

Use `--reset` only when you intentionally want to delete and recreate the
configured Chroma collection.

Query the local vector index:

```bash
python scripts/query_vector_index.py "What does the EYLF say about play-based learning?" --top-k 5
```

Query with BM25 or hybrid retrieval:

```bash
python scripts/query_vector_index.py "play based learning" --mode bm25 --top-k 5
python scripts/query_vector_index.py "play based learning" --mode hybrid --reranker cross_encoder --top-k 5
```

Run the policy RAG main-graph smoke test:

```bash
python scripts/policy_rag_smoke_test.py
python scripts/policy_rag_smoke_test.py --real-model --reranker cross_encoder
```

Run the deterministic Week 2 RAG and memory evaluation set (15 policy
retrieval/citation cases and 5 memory-boundary cases):

```bash
python scripts/run_week2_evals.py
```

The evaluation uses BM25 plus the tracked knowledge chunks and a temporary
SQLite database. It does not call an external model. A non-zero exit code
means at least one expected retrieval, clarification, or memory boundary failed.

Run real LLM policy RAG and print the final answer:

```bash
python scripts/real_policy_rag_answer.py "What does the EYLF say about play-based learning?"
```

## Local Data

Ignored local runtime data:

```text
.env
data/local/
data/chroma/
```

Tracked knowledge inputs and processed chunks live under:

```text
data/knowledge/
```
