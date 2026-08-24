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
from app.services.retry import RetryPolicy
from app.services.observation_redactor import ObservationRedactor
from app.services.request_guard import (
    EasyTeachingRequestGuard,
    RequestGuardAction,
    RequestGuardResult,
)
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
from app.services.async_store import AsyncEasyTeachingStore, ConversationSessionBusyError
from app.services.knowledge_ingestion import (
    KnowledgeIngestionService,
    KnowledgeSourceSpec,
    ParsedTextBlock,
)
from app.services.knowledge_retriever import (
    CrossEncoderReranker,
    KnowledgeRetriever,
)
from app.services.embedding_provider import EmbeddingResponse, GeminiEmbeddingProvider
from app.services.vector_store import ChromaVectorStore, VectorIndexConfigurationError
from app.services.lexical_index import (
    LexicalIndexConfigurationError,
    SQLiteFTS5KnowledgeIndex,
)

__all__ = [
    "ChatCompletionsModelProvider",
    "ChromaVectorStore",
    "ContextManager",
    "ConversationMemoryUpdate",
    "ConversationSessionBusyError",
    "CrossEncoderReranker",
    "AsyncEasyTeachingStore",
    "EmbeddingResponse",
    "GeminiEmbeddingProvider",
    "KnowledgeIngestionService",
    "KnowledgeRetriever",
    "LexicalIndexConfigurationError",
    "LLMContextSummarizer",
    "LLMLongTermMemoryExtractor",
    "KnowledgeSourceSpec",
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
    "EasyTeachingRequestGuard",
    "RequestGuardAction",
    "RequestGuardResult",
    "ParsedTextBlock",
    "SQLiteFTS5KnowledgeIndex",
    "RetryPolicy",
    "VectorIndexConfigurationError",
]
