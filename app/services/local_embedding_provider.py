"""Lazy local embeddings for private, teacher-scoped knowledge."""

import asyncio
from typing import Any, List


class LocalSentenceTransformerEmbeddingProvider:
    """Keep uploaded centre-document text on the local machine."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: Any = None

    async def embed_texts_async(
        self,
        texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        del task_type
        return await asyncio.to_thread(self._embed_texts, texts)

    async def embed_text_async(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> List[float]:
        del task_type
        return (await asyncio.to_thread(self._embed_texts, [text]))[0]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts must contain at least one item")
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = [[float(value) for value in vector] for vector in vectors]
        if any(len(vector) != self.dimension for vector in result):
            raise ValueError(
                "Local embedding dimension does not match the configured scoped index"
            )
        return result
