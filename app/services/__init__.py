"""Shared service integrations."""

from app.services.model_errors import (
    ModelConfigurationError,
    ModelErrorCode,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelProviderError,
    ModelTimeoutError,
)
from app.services.model_provider import ChatCompletionsModelProvider
from app.services.learning_record_drafting import LearningRecordDraftingService
from app.services.observation_redactor import ObservationRedactor
from app.services.model_types import (
    ModelJSONParseResult,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
from app.services.context_manager import ContextManager
from app.services.context_summarizer import ConversationMemoryUpdate, LLMContextSummarizer
from app.services.long_memory_extractor import LLMLongTermMemoryExtractor
from app.services.policy_rag import PolicyRAGService
from app.services.store import EduFlowStore
from app.services.knowledge_ingestion import (
    KnowledgeIngestionService,
    KnowledgeSourceSpec,
    ParsedTextBlock,
)
from app.services.knowledge_retriever import (
    BM25KnowledgeIndex,
    CrossEncoderReranker,
    KnowledgeRetriever,
    LexicalReranker,
)
from app.services.embedding_provider import EmbeddingResponse, GeminiEmbeddingProvider
from app.services.vector_store import ChromaVectorStore, VectorIndexConfigurationError

__all__ = [
    "ChatCompletionsModelProvider",
    "ChromaVectorStore",
    "ContextManager",
    "ConversationMemoryUpdate",
    "CrossEncoderReranker",
    "EduFlowStore",
    "EmbeddingResponse",
    "GeminiEmbeddingProvider",
    "BM25KnowledgeIndex",
    "KnowledgeIngestionService",
    "KnowledgeRetriever",
    "LexicalReranker",
    "LLMContextSummarizer",
    "LLMLongTermMemoryExtractor",
    "KnowledgeSourceSpec",
    "LearningRecordDraftingService",
    "ModelConfigurationError",
    "ModelErrorCode",
    "ModelHTTPError",
    "ModelInvalidResponseError",
    "ModelJSONParseResult",
    "ModelMessage",
    "ModelProviderError",
    "ModelRequest",
    "ModelResponse",
    "ModelRole",
    "ModelTimeoutError",
    "ModelUsage",
    "ObservationRedactor",
    "ParsedTextBlock",
    "PolicyRAGService",
    "VectorIndexConfigurationError",
]
