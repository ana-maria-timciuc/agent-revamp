"""Preprocessing pipeline: normalize message -> classify intent -> retrieve skills/tools.

This is the diagram's whole "Preprocess" box as one entrypoint: Agent.chat() (and the
CLI debug path in main.py::pipeline_demo) call `PreprocessPipeline.run()` once per turn
and get back a ContextPackage with the classified intent plus reranked skill/tool
candidates, ready to fold into the prompt/tool list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent_revamp.config import settings
from agent_revamp.preprocess.catalog import CatalogEntry, QdrantCatalog
from agent_revamp.preprocess.embeddings import EmbeddingService, OpenAIEmbeddingService
from agent_revamp.preprocess.intent import IntentClassifier, IntentResult, LLMIntentClassifier


@dataclass
class ContextPackage:
    intent: IntentResult
    skills: list[CatalogEntry] = field(default_factory=list)
    tools: list[CatalogEntry] = field(default_factory=list)
    raw_query: str = ""
    skills_passed: bool = True
    tools_passed: bool = True


def preprocess_text(text: str) -> str:
    """Light normalization before classification/retrieval: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip())


class PreprocessPipeline:
    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        catalog: QdrantCatalog | None = None,
        embedder: EmbeddingService | None = None,
        openai_client: Any | None = None,
    ):
        if classifier is None:
            from openai import AsyncOpenAI

            client = openai_client or AsyncOpenAI(
                api_key=settings.openai_key, base_url=settings.get_openai_base_url()
            )
            classifier = LLMIntentClassifier(client=client)
        self.classifier = classifier
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
            skills_passed=result.skills_passed,
            tools_passed=result.tools_passed,
        )

    async def close(self) -> None:
        await self.catalog.close()
