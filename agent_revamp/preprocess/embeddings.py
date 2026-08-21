"""Embedding service abstraction for the RAG skill/tool retrieval pipeline.

Exposes a small ABC so the vectorizer can be swapped (OpenAI today, a local/finetuned
model later) without touching the catalog or the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_revamp.config import settings


class EmbeddingService(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality — must match the Qdrant collection config."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/queries into vectors of `dimension` size."""

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


class GemmaEmbeddingService(EmbeddingService):
    """Placeholder for a local embeddinggemma-300m backend — not implemented yet."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model

    @property
    def dimension(self) -> int:
        return {"embeddinggemma-300m": 768}.get(self.model, 768)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("GemmaEmbeddingService.embed_texts is not implemented yet")

    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError("GemmaEmbeddingService.embed_text is not implemented yet")


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model
        from openai import AsyncOpenAI

        self._client: Any = AsyncOpenAI(
            api_key=settings.openai_key, base_url=settings.get_openai_base_url()
        )

    @property
    def dimension(self) -> int:
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }.get(self.model, 1536)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
