"""Intent classification for the preprocessing pipeline.

Abstract behind an ABC so the LLM-based taxonomy classifier used today can be swapped
for a trained classification model later without touching the pipeline. The classifier
only ever produces a label from a fixed taxonomy plus a confidence score.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_revamp.config import settings


@dataclass
class IntentResult:
    intent: str
    confidence: float
    raw_label: str
    taxonomy_version: str = "v1"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.6


class IntentClassifier(ABC):
    @abstractmethod
    async def classify(self, text: str) -> IntentResult:
        """Classify a preprocessed user message into an intent label."""


class LLMIntentClassifier(IntentClassifier):
    """Cheap single OpenAI call that maps a message onto a fixed taxonomy label.

    The taxonomy is configurable via INTENT_TAXONOMY (comma-separated labels) and is
    frozen into the prompt, so labels stay stable across model swaps.
    """

    def __init__(self, taxonomy: str | None = None):
        self.taxonomy = (taxonomy or settings.intent_taxonomy).strip()
        if not self.taxonomy:
            raise ValueError("intent taxonomy must not be empty")

    async def classify(self, text: str) -> IntentResult:
        from agent_revamp.agent import _get_openai

        labels = ", ".join(
            label.strip() for label in self.taxonomy.split(",") if label.strip()
        )
        prompt = (
            "You are an intent router. Classify the user's message into EXACTLY ONE of these "
            f"intents: {labels}.\n"
            "Rules:\n"
            "- analytics: analyzing, summarizing, or comparing platform data (no chart needed)\n"
            "- reporting: the user should SEE a visual chart/KPI report\n"
            "- lookup: fetching a specific record or value (a property, a transaction, a name->id)\n"
            "- create_record: adding/creating/updating data (transaction, asset, entity, loan, ...)\n"
            "- market_research: external/market data, web research, regulations, costs estimates\n"
            "- meta_about: questions about the assistant itself, capabilities, or how to use it\n"
            "- off_domain: anything outside the platform's scope\n"
            'Respond with a JSON object only: {"intent": <label>, "confidence": <0..1>}\n'
        )
        response = await _get_openai().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:2000]},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            intent = str(data.get("intent", "")).strip().lower()
            confidence = float(data.get("confidence", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            intent, confidence = "", 0.0
        return IntentResult(intent=intent, confidence=confidence, raw_label=raw)
