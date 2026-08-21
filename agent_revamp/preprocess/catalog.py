"""Qdrant-backed catalog of skills and tools for RAG retrieval — the diagram's "RAG" box.

Merges what used to be two separate implementations:
- The structure (CatalogEntry/RetrievalResult, pluggable EmbeddingService + Reranker,
  two collections) comes from the original preprocessing-pipeline scaffold.
- The robustness (every public method degrades gracefully instead of raising, so a
  downed Qdrant never crashes a turn) comes from the implementation that was actually
  wired into Agent.

On top of that, this module owns the reranker fail->retry loop from the architecture
diagram (Reranker --fail--> RAG, widening top_k and re-querying, bounded by
settings.rerank_max_retries), and generalizes orphan-pruning (previously skills-only)
to both kinds, scoped per `agent` (process class) so one process class's reindex can
never delete another's entries out of the shared collections.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from agent_revamp.config import settings
from agent_revamp.preprocess.embeddings import EmbeddingService
from agent_revamp.preprocess.reranker import RankedHit, Reranker, ScoreReranker

logger = logging.getLogger(__name__)

KIND_SKILL = "skill"
KIND_TOOL = "tool"

_ID_NAMESPACE = uuid.NAMESPACE_URL


def _point_id(entry_id: str) -> str:
    """Deterministic UUID point id from a human-readable entry id (Qdrant requires
    unsigned ints or UUIDs as point ids; we keep the readable id in the payload)."""
    return str(uuid.uuid5(_ID_NAMESPACE, entry_id))


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
    skills_passed: bool = True
    tools_passed: bool = True

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
        self.skills_collection = skills_collection or settings.qdrant_skills_collection
        self.tools_collection = tools_collection or settings.qdrant_tools_collection
        self.reranker = reranker or ScoreReranker()
        self.top_k = top_k or settings.qdrant_top_k
        self._client: AsyncQdrantClient | None = None

    async def _client_or_create(self) -> AsyncQdrantClient | None:
        if self._client is not None:
            return self._client
        try:
            self._client = AsyncQdrantClient(url=self.url)
        except Exception as exc:
            logger.warning("Could not create Qdrant client: %s", exc)
            return None
        return self._client

    async def _ensure_collection(self, client: AsyncQdrantClient, name: str) -> bool:
        dim = self.embedder.dimension
        try:
            exists = await client.collection_exists(name)
            if exists:
                info = await client.get_collection(name)
                vectors = getattr(info.config.params, "vectors", None)
                current_dim = vectors.get("size") if isinstance(vectors, dict) else getattr(vectors, "size", None)
                if current_dim != dim:
                    logger.warning("Recreating collection %s (dim %s -> %s)", name, current_dim, dim)
                    await client.delete_collection(name)
                    exists = False
            if not exists:
                await client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
            return True
        except Exception:
            logger.warning("Could not ensure Qdrant collection %s", name)
            return False

    def _collection_for(self, kind: str) -> str:
        return self.skills_collection if kind == KIND_SKILL else self.tools_collection

    async def upsert(self, entries: list[CatalogEntry], prune: bool = True) -> bool:
        """Embed + upsert entries (mixed kinds/agents in one call are fine — grouped
        internally). Never raises: any embedding or Qdrant failure returns False,
        mirroring the graceful-degradation contract Agent.__aenter__ depends on."""
        if not entries:
            return True
        client = await self._client_or_create()
        if client is None:
            return False

        by_collection: dict[str, list[CatalogEntry]] = {}
        for entry in entries:
            by_collection.setdefault(self._collection_for(entry.kind), []).append(entry)

        try:
            for collection, group in by_collection.items():
                if not await self._ensure_collection(client, collection):
                    return False
                vectors = await self.embedder.embed_texts([e.content for e in group])
                await client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(id=_point_id(e.id), vector=vec, payload=e.to_payload())
                        for e, vec in zip(group, vectors, strict=False)
                    ],
                )
                if prune:
                    await self._prune_orphans(client, collection, group)
            return True
        except Exception as exc:
            logger.warning("Qdrant upsert failed: %s", exc)
            return False

    async def _prune_orphans(self, client: AsyncQdrantClient, collection: str, group: list[CatalogEntry]) -> None:
        """Delete stale points (renamed/removed entries) — scoped to each entry's own
        `agent` (process class), so reindexing one process class's entries never
        touches another's points sharing the same collection."""
        by_agent: dict[str, set[str]] = {}
        for entry in group:
            by_agent.setdefault(entry.agent, set()).add(_point_id(entry.id))
        for agent_name, keep_ids in by_agent.items():
            try:
                points, _ = await client.scroll(
                    collection_name=collection,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="agent", match=models.MatchValue(value=agent_name))]
                    ),
                    limit=1000,
                )
                orphans = [str(p.id) for p in points if str(p.id) not in keep_ids]
                if orphans:
                    logger.info(
                        "Pruning %d stale point(s) for agent=%s from %s", len(orphans), agent_name, collection
                    )
                    await client.delete(collection_name=collection, points_selector=orphans)
            except Exception as exc:
                logger.warning("Orphan cleanup failed for %s/agent=%s: %s", collection, agent_name, exc)

    @staticmethod
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

    async def _search_kind(
        self, client: AsyncQdrantClient, query: str, vector: list[float], kind: str, collection: str, top_k: int
    ) -> tuple[list[tuple[CatalogEntry, float]], bool]:
        """The RAG-box's fail->retry loop: rerank the current hits, and on `passed=False`
        widen top_k and re-query, up to settings.rerank_max_retries times. Always returns
        the best-effort (last attempt's) hits — a turn is never blocked indefinitely."""
        k = top_k
        last_hits: list[tuple[CatalogEntry, float]] = []
        max_retries = settings.rerank_max_retries
        for attempt in range(max_retries + 1):
            try:
                points = await client.query_points(collection_name=collection, query=vector, limit=k, with_payload=True)
            except Exception as exc:
                logger.warning("Qdrant search failed (%s): %s", kind, exc)
                return last_hits, False
            ranked = [RankedHit(entry, score) for entry, score in self._to_entries(points.points, kind)]
            outcome = await self.reranker.rerank(query, ranked)
            last_hits = [(h.entry, h.score) for h in outcome.hits]
            if outcome.passed or attempt == max_retries:
                return last_hits, outcome.passed
            k *= settings.rerank_retry_top_k_multiplier
        return last_hits, False

    async def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Never raises. Two distinct degrade-gracefully outcomes, both safe for callers
        to treat as "no candidates, proceed without them":
        - No client available, or the query couldn't even be embedded -> empty result
          with passed=True (default) — we never got far enough to say retrieval failed.
        - A live query_points call fails (e.g. Qdrant reachable at connect time but the
          server dies mid-session) -> passed=False, surfaced via _search_kind, so callers
          that care (e.g. the --preprocess debug output) can distinguish "nothing to find"
          from "retrieval broke."
        """
        client = await self._client_or_create()
        if client is None:
            return RetrievalResult(skills=[], tools=[])
        try:
            query_vector = await self.embedder.embed_text(query)
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return RetrievalResult(skills=[], tools=[])

        k = top_k or self.top_k
        skill_hits, skills_passed = await self._search_kind(
            client, query, query_vector, KIND_SKILL, self.skills_collection, k
        )
        tool_hits, tools_passed = await self._search_kind(
            client, query, query_vector, KIND_TOOL, self.tools_collection, k
        )
        return RetrievalResult(skills=skill_hits, tools=tool_hits, skills_passed=skills_passed, tools_passed=tools_passed)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
