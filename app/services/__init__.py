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
from app.services.model_types import (
    ModelJSONParseResult,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
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
    "CrossEncoderReranker",
    "EduFlowStore",
    "EmbeddingResponse",
    "GeminiEmbeddingProvider",
    "BM25KnowledgeIndex",
    "KnowledgeIngestionService",
    "KnowledgeRetriever",
    "LexicalReranker",
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
    "ParsedTextBlock",
    "PolicyRAGService",
    "VectorIndexConfigurationError",
]
