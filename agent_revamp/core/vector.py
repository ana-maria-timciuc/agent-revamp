"""Qdrant-backed index of agent skills and tools (the "Dynamic skills, tools" node).

- ToolIndex: embeds each MCP tool (name + description) and upserts it into the
  tools collection, point id = tool name, so re-indexing is idempotent.
- SkillIndex: each skill file is one entry — one point per skill, point id =
  skill name, so re-indexing never duplicates and renaming/removing a skill
  prunes its stale point.

Every public method degrades gracefully: index/query failures return None/False
instead of raising, so the agent keeps working (with all tools / no skill
context) when Qdrant is down.
"""

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from agent_revamp.config import settings

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.NAMESPACE_DNS


def _point_id(key: str) -> str:
    """Qdrant point ids must be integers or UUIDs — derive a stable UUID from a string key."""
    return str(uuid.uuid5(_ID_NAMESPACE, key))


class Embedder:
    """OpenAI text embeddings, batched."""

    def __init__(self, client: Any, model: str | None = None):
        self._client = client
        self.model = model or settings.qdrant_embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            resp = await self._client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None


class VectorIndex:
    def __init__(
        self,
        url: str | None = None,
        collection: str = "agent",
        embedder: Embedder | None = None,
        force_recreate: bool = False,
    ):
        self._url = url or settings.qdrant_url
        self.collection = collection
        self.embedder = embedder
        self._client: AsyncQdrantClient | None = None
        self._dim = 0
        self._force_recreate = force_recreate

    async def __aenter__(self) -> "VectorIndex":
        self._client = AsyncQdrantClient(url=self._url)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.close()

    async def _ensure_collection(self, dim: int) -> bool:
        assert self._client is not None
        try:
            exists = await self._client.collection_exists(self.collection)
            if exists:
                info = await self._client.get_collection(self.collection)
                vectors = getattr(info.config.params, "vectors", None)
                current_dim = vectors.get("size") if isinstance(vectors, dict) else getattr(vectors, "size", None)
                if self._force_recreate or current_dim != dim:
                    logger.warning(
                        "Recreating collection %s (dim %s -> %s)", self.collection, current_dim, dim
                    )
                    await self._client.delete_collection(self.collection)
                    exists = False
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception:
            logger.warning("Could not ensure Qdrant collection %s", self.collection)
            return False
        return True

    async def upsert(self, texts: list[str], point_ids: list[str], payloads: list[dict] | None = None) -> bool:
        """Embed texts and upsert with the given string point ids. Returns False on any failure."""
        if self._client is None:
            return False
        vectors = await self.embedder.embed(texts)
        if vectors is None:
            return False
        if not vectors:
            return True
        if not await self._ensure_collection(len(vectors[0])):
            return False
        try:
            await self._client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=pid,
                        vector=vec,
                        payload=payloads[i] if payloads else {"text": texts[i]},
                    )
                    for i, (pid, vec) in enumerate(zip(point_ids, vectors))
                ],
            )
            return True
        except Exception as exc:
            logger.warning("Qdrant upsert failed: %s", exc)
            return False

    async def search(self, query: str, top_k: int = 8) -> list[dict] | None:
        """Vector-search for the query. Returns [{id, text, score, ...payload}], or None on failure."""
        if self._client is None:
            return None
        vector = (await self.embedder.embed([query])) or [None]
        if vector[0] is None:
            return None
        try:
            points = await self._client.query_points(
                collection_name=self.collection,
                query=vector[0],
                limit=top_k,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return None
        return [
            {
                "id": p.id,
                "score": p.score,
                **{k: v for k, v in (p.payload or {}).items()},
            }
            for p in points.points
        ]


class ToolIndex(VectorIndex):
    """Indexes MCP tool definitions; search returns matching tool names."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("collection", settings.qdrant_tools_collection)
        super().__init__(*args, **kwargs)

    async def index_tools(self, tools: list[Any]) -> bool:
        texts = [f"{t.name}\n{t.description or ''}" for t in tools]
        payloads = [{"tool_name": t.name, "description": t.description or ""} for t in tools]
        return await self.upsert(texts, [_point_id(t.name) for t in tools], payloads)

    async def search_tools(self, query: str, top_k: int | None = None) -> list[dict] | None:
        hits = await self.search(query, top_k or settings.qdrant_top_k)
        if hits is None:
            return None
        return [h for h in hits if h.get("tool_name")]


class SkillIndex(VectorIndex):
    """Indexes skills; each skill is one entry (one point). Search returns matching skills."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("collection", settings.qdrant_skills_collection)
        super().__init__(*args, **kwargs)

    async def index_skills(self, skills: list[tuple[str, str]]) -> bool:
        """Upsert one point per (skill_name, text) pair. Points whose skill no longer
        exists (renamed/removed) are pruned, so stale entries never come back in search."""
        if not skills:
            return True
        names = [name for name, _ in skills]
        texts = [text for _, text in skills]
        ids = [_point_id(name) for name in names]
        payloads = [{"skill_name": name, "text": text} for name, text in skills]
        ok = await self.upsert(texts, ids, payloads)
        if ok:
            await self._prune_orphans(set(ids))
        return ok

    async def _prune_orphans(self, keep_ids: set[str]) -> None:
        if self._client is None:
            return
        try:
            points, _ = await self._client.scroll(collection_name=self.collection, limit=1000)
            orphans = [str(p.id) for p in points if str(p.id) not in keep_ids]
            if orphans:
                logger.info("Pruning %d stale skill point(s) from %s", len(orphans), self.collection)
                await self._client.delete(collection_name=self.collection, points_selector=orphans)
        except Exception as exc:
            logger.warning("Skill orphan cleanup failed: %s", exc)

    async def search_skills(self, query: str, top_k: int | None = None) -> list[str] | None:
        hits = await self.search(query, top_k or settings.qdrant_top_k)
        if hits is None:
            return None
        return [h.get("text", "") for h in hits if h.get("text")]
