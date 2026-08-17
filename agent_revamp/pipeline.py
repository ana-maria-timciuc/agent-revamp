"""Preprocessing pipeline: normalize message → classify intent → retrieve skills/tools.

Produces an abstract ContextPackage (intent + retrieved skills + retrieved tools) that the
later model-consumption stage can shape however it needs. The structure is intentionally
loose right now — the model-input contract is still under discussion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_revamp.catalog import CatalogEntry, QdrantCatalog
from agent_revamp.embeddings import EmbeddingService, OpenAIEmbeddingService
from agent_revamp.intent import IntentClassifier, IntentResult, LLMIntentClassifier


@dataclass
class ContextPackage:
    intent: IntentResult
    skills: list[CatalogEntry] = field(default_factory=list)
    tools: list[CatalogEntry] = field(default_factory=list)
    raw_query: str = ""


def preprocess_text(text: str) -> str:
    """Light normalization before classification/retrieval: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip())


class PreprocessPipeline:
    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        catalog: QdrantCatalog | None = None,
        embedder: EmbeddingService | None = None,
    ):
        self.classifier = classifier or LLMIntentClassifier()
        if catalog is None:
            embedder = embedder or OpenAIEmbeddingService()
            catalog = QdrantCatalog(embedder=embedder)
        self.catalog = catalog

    async def run(self, user_message: str) -> ContextPackage:
        raw_query = preprocess_text(user_message)
        intent = await self.classifier.classify(raw_query)
        result = await self.catalog.search(raw_query)
        return ContextPackage(
            intent=intent,
            skills=result.skill_entries(),
            tools=result.tool_entries(),
            raw_query=raw_query,
        )

    async def close(self) -> None:
        await self.catalog.close()
