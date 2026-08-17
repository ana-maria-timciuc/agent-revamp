"""Embedding service abstraction for the RAG skill/tool retrieval pipeline.

Exposes a small ABC so the vectorizer can be swapped (OpenAI today, a local/finetuned
model later) without touching the catalog or the pipeline. The OpenAI implementation
reuses the existing OPENAI_KEY / base-url config from agent.py.
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
