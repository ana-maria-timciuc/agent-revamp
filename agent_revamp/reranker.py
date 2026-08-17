"""Reranking abstraction for RAG retrieval.

Currently a score-based default that just re-sorts by the raw vector score. The ABC
allows a proper cross-encoder reranker (e.g. a sentence-transformer or a dedicated
model) to be dropped in later without touching the catalog or pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RankedHit:
    entry: Any
    score: float


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        """Re-order hits by relevance to the query, best first."""


class ScoreReranker(Reranker):
    """No-op semantic rerank — keeps the vector-score ordering (descending)."""

    async def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        return sorted(hits, key=lambda h: h.score, reverse=True)
