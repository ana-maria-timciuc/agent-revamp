"""Reranking abstraction for RAG retrieval.

Deliberately left abstract: the concrete backend (cross-encoder, multi-vector, an
LLM-judge, or a custom scorer) is still being chosen. `ScoreReranker` is a trivial
default that keeps today's behavior (vector-score order, "succeeded" whenever there's
at least one hit) so the pipeline works end-to-end while a real reranker is decided.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RankedHit:
    entry: Any
    score: float


@dataclass
class RerankOutcome:
    hits: list[RankedHit]
    passed: bool


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, hits: list[RankedHit]) -> RerankOutcome:
        """Re-order hits by relevance to the query (best first) and report whether the
        result is good enough to use (`passed`). Callers retry retrieval on `passed=False`."""


class ScoreReranker(Reranker):
    """No-op semantic rerank — keeps the vector-score ordering (descending), and
    considers the result a pass whenever there's at least one hit."""

    async def rerank(self, query: str, hits: list[RankedHit]) -> RerankOutcome:
        ordered = sorted(hits, key=lambda h: h.score, reverse=True)
        return RerankOutcome(hits=ordered, passed=bool(ordered))
