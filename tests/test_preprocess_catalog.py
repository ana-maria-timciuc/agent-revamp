"""Unit tests for the merged RAG box, agent_revamp/preprocess/catalog.py::QdrantCatalog.
No live Qdrant needed -- exercises graceful degradation (unreachable server), the
reranker fail->retry loop (bounded, widening top_k), and per-agent-scoped orphan
pruning, all against fakes.
"""

import asyncio

from agent_revamp.config import settings
from agent_revamp.preprocess.catalog import (
    KIND_SKILL,
    KIND_TOOL,
    CatalogEntry,
    QdrantCatalog,
    _point_id,
)
from agent_revamp.preprocess.embeddings import EmbeddingService
from agent_revamp.preprocess.reranker import Reranker, RerankOutcome


def _run(coro):
    return asyncio.run(coro)


class _FakeEmbedder(EmbeddingService):
    @property
    def dimension(self) -> int:
        return 4

    async def embed_texts(self, texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class _FailAlwaysReranker(Reranker):
    """Never passes -- used to prove the retry loop is bounded, not to model any real
    reranking strategy (the real one is still being chosen -- see reranker.py)."""

    def __init__(self):
        self.calls = 0

    async def rerank(self, query, hits):
        self.calls += 1
        return RerankOutcome(hits=hits, passed=False)


def test_upsert_and_search_degrade_gracefully_when_qdrant_is_unreachable():
    catalog = QdrantCatalog(embedder=_FakeEmbedder(), url="http://127.0.0.1:1")  # nothing listening
    ok = _run(catalog.upsert([CatalogEntry(id="t1", kind=KIND_TOOL, name="t1", content="hello", agent="penny")]))
    assert ok is False

    result = _run(catalog.search("hello"))
    assert result.skills == [] and result.tools == []
    # The client is constructed lazily and only fails on a real RPC call, so an
    # unreachable server surfaces as a query_points failure -> passed=False (distinct
    # from "no client at all"/"embedding failed" -> passed=True, see catalog.py::search).
    assert result.skills_passed is False and result.tools_passed is False
    _run(catalog.close())


def test_unreachable_qdrant_short_circuits_before_ever_calling_the_reranker():
    reranker = _FailAlwaysReranker()
    catalog = QdrantCatalog(embedder=_FakeEmbedder(), url="http://127.0.0.1:1", reranker=reranker)
    _run(catalog.search("hello"))
    assert reranker.calls == 0


def test_reranker_retry_loop_is_bounded_and_widens_top_k():
    class _FakeClient:
        def __init__(self):
            self.limits_seen = []

        async def query_points(self, collection_name, query, limit, with_payload):
            self.limits_seen.append(limit)
            return SimpleNamespaceLike(points=[])

        async def close(self):
            pass

    class SimpleNamespaceLike:
        def __init__(self, points):
            self.points = points

    fake_client = _FakeClient()
    reranker = _FailAlwaysReranker()
    catalog = QdrantCatalog(embedder=_FakeEmbedder(), reranker=reranker, top_k=3)
    catalog._client = fake_client  # bypass real AsyncQdrantClient construction

    hits, passed = _run(catalog._search_kind(fake_client, "q", [0, 0, 0, 0], KIND_TOOL, "agent_tools", 3))
    assert passed is False
    assert reranker.calls == settings.rerank_max_retries + 1
    expected_limits = [3 * (settings.rerank_retry_top_k_multiplier**i) for i in range(settings.rerank_max_retries + 1)]
    assert fake_client.limits_seen == expected_limits
    assert hits == []


def test_orphan_pruning_is_scoped_per_agent():
    """Regression guard for the exact reason upsert() prunes per-agent rather than
    globally per-collection: dollar_bill/uncle_sam share the same collections as penny,
    so a penny reindex must never delete another process class's points."""

    class _FakeClient:
        def __init__(self):
            self.deleted = None

        async def collection_exists(self, name):
            return False  # forces create_collection path, no get_collection() needed

        async def create_collection(self, collection_name, vectors_config):
            pass

        async def upsert(self, collection_name, points):
            pass

        async def scroll(self, collection_name, scroll_filter, limit):
            agent_filter = scroll_filter.must[0].match.value
            if agent_filter == "penny":
                return [_Point("stale-penny-point"), _Point("keep-me")], None
            return [_Point("dollar-bill-point-should-survive")], None

        async def delete(self, collection_name, points_selector):
            self.deleted = points_selector

        async def close(self):
            pass

    class _Point:
        def __init__(self, id):
            self.id = id

    fake_client = _FakeClient()
    catalog = QdrantCatalog(embedder=_FakeEmbedder())
    catalog._client = fake_client

    entry = CatalogEntry(id="keep-me-src", kind=KIND_SKILL, name="keep-me", content="x", agent="penny")
    import agent_revamp.preprocess.catalog as catalog_module

    original_point_id = catalog_module._point_id
    try:
        catalog_module._point_id = lambda eid: "keep-me" if eid == "keep-me-src" else original_point_id(eid)
        _run(catalog.upsert([entry], prune=True))
    finally:
        catalog_module._point_id = original_point_id

    assert fake_client.deleted == ["stale-penny-point"]


def test_point_id_is_stable_and_deterministic():
    assert _point_id("execute_query") == _point_id("execute_query")
    assert _point_id("execute_query") != _point_id("generate_report")
