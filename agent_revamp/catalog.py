"""Qdrant-backed catalog of skills and tools for RAG retrieval.

Two collections (skills, tools). Entries are indexed as CatalogEntry records; the
`payload` dict is free-form so the pipeline can hand back whatever the model-consumption
stage needs (full skill text for skills; OpenAI tool schema for tools).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from agent_revamp.config import settings
from agent_revamp.embeddings import EmbeddingService
from agent_revamp.reranker import RankedHit, Reranker, ScoreReranker

KIND_SKILL = "skill"
KIND_TOOL = "tool"


def _point_id(entry_id: str) -> str:
    """Deterministic UUID point id from a human-readable entry id (Qdrant requires
    unsigned ints or UUIDs as point ids; we keep the readable id in the payload)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, entry_id))


@dataclass
class CatalogEntry:
    id: str
    kind: str
    name: str
    content: str
    agent: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "entry_id": self.id,
            "kind": self.kind,
            "name": self.name,
            "content": self.content,
            "agent": self.agent,
            **self.payload,
        }


@dataclass
class RetrievalResult:
    skills: list[tuple[CatalogEntry, float]]
    tools: list[tuple[CatalogEntry, float]]

    def skill_entries(self) -> list[CatalogEntry]:
        return [e for e, _ in self.skills]

    def tool_entries(self) -> list[CatalogEntry]:
        return [e for e, _ in self.tools]


class QdrantCatalog:
    def __init__(
        self,
        embedder: EmbeddingService,
        url: str | None = None,
        skills_collection: str | None = None,
        tools_collection: str | None = None,
        reranker: Reranker | None = None,
        top_k: int | None = None,
    ):
        self.embedder = embedder
        self.url = url or settings.qdrant_url
        self.skills_collection = skills_collection or settings.skills_collection
        self.tools_collection = tools_collection or settings.tools_collection
        self.reranker = reranker or ScoreReranker()
        self.top_k = top_k or settings.retrieval_top_k
        self._client: AsyncQdrantClient | None = None

    async def _client_or_create(self) -> AsyncQdrantClient:
        if self._client is not None:
            return self._client
        client = AsyncQdrantClient(url=self.url)
        await self._ensure_collection(client, self.skills_collection)
        await self._ensure_collection(client, self.tools_collection)
        self._client = client
        return client

    async def _ensure_collection(self, client: AsyncQdrantClient, name: str) -> None:
        exists = await client.collection_exists(name)
        if not exists:
            await client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert(self, entries: list[CatalogEntry]) -> None:
        client = await self._client_or_create()
        for entry in entries:
            vector = await self.embedder.embed_text(entry.content)
            await client.upsert(
                collection_name=(
                    self.skills_collection
                    if entry.kind == KIND_SKILL
                    else self.tools_collection
                ),
                points=[
                    models.PointStruct(
                        id=_point_id(entry.id),
                        vector=vector,
                        payload=entry.to_payload(),
                    )
                ],
            )

    async def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        client = await self._client_or_create()
        k = top_k or self.top_k
        query_vector = await self.embedder.embed_text(query)

        def _to_entries(hits: list[Any], kind: str) -> list[tuple[CatalogEntry, float]]:
            results = []
            for hit in hits:
                payload = hit.payload or {}
                entry = CatalogEntry(
                    id=payload.get("entry_id", str(hit.id)),
                    kind=kind,
                    name=payload.get("name", ""),
                    content=payload.get("content", ""),
                    agent=payload.get("agent", ""),
                    payload={
                        k: v
                        for k, v in payload.items()
                        if k not in {"entry_id", "kind", "name", "content", "agent"}
                    },
                )
                results.append((entry, float(hit.score)))
            return results

        skill_hits = await client.query_points(
            collection_name=self.skills_collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        tool_hits = await client.query_points(
            collection_name=self.tools_collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        skills = await self.reranker.rerank(
            query,
            [RankedHit(e, s) for e, s in _to_entries(skill_hits.points, KIND_SKILL)],
        )
        tools = await self.reranker.rerank(
            query,
            [RankedHit(e, s) for e, s in _to_entries(tool_hits.points, KIND_TOOL)],
        )

        return RetrievalResult(
            skills=[(h.entry, h.score) for h in skills],
            tools=[(h.entry, h.score) for h in tools],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
