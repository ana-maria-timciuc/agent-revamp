"""Postprocess response validation — the diagram's "Validation" sub-box inside the
Postprocess zone, deciding Yes (return the answer) vs No (loop back to the Agent for
another attempt). Not to be confused with the *other* "Validate tools" box between MCP
and the model (preprocess/process_class.py::validate_tool_scope/filter_tools) — that one
guards which tools the agent may call; this one judges the agent's finished answer.

Two checks, cheapest and most decisive first:
1. `check_leak_safety` — deterministic, free. If postprocess/leak_guard.py's
   `sanitize_user_text()` had to change the model's raw text at all, the model attempted
   to reveal internal schema/SQL/product names — an automatic fail. No need to also spend
   an LLM call confirming what the sanitizer already caught.
2. `check_quality` — one extra LLM call, only made when the leak-safety check passed.
   Asks whether the (already-sanitized) answer is a genuine, on-topic attempt at the
   user's question. This is a quality gate, not a security boundary — leak_guard already
   guarantees the text is safe to show regardless of this check's outcome — so any
   failure to reach/parse the judge fails OPEN (treated as passed) rather than blocking
   the turn.

Constructor-injected OpenAI client, same pattern as preprocess/intent.py's
LLMIntentClassifier, so this module has zero import-time dependency on agent.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    reason: str


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA reviewer for a bookkeeping assistant. Given the user's question "
    "and the assistant's proposed answer, decide whether the answer is a genuine, on-topic "
    "attempt at addressing the question — not whether it is perfectly worded, just whether "
    "it is a real attempt rather than a non-answer, refusal, or empty/off-topic reply.\n\n"
    'Respond with strict JSON only: {"passed": true|false, "reason": "<one short sentence>"}.'
)


class ResponseValidator:
    def __init__(self, client: AsyncOpenAI, model: str | None = None):
        self._client = client
        self._model = model

    def check_leak_safety(self, raw_text: str, sanitized_text: str) -> ValidationResult:
        """Free, deterministic: sanitize_user_text() is a no-op on clean text, so any
        difference means it caught something that needed redacting or substituting."""
        if raw_text != sanitized_text:
            return ValidationResult(
                passed=False,
                reason="response contained internal schema/SQL details that had to be redacted",
            )
        return ValidationResult(passed=True, reason="")

    async def check_quality(self, user_message: str, answer_text: str, model: str) -> ValidationResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model or model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {user_message}\n\nAnswer: {answer_text}"},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return ValidationResult(passed=bool(data.get("passed", True)), reason=str(data.get("reason", "")))
        except Exception as exc:
            logger.warning("Quality judge call failed, failing open: %s", exc)
            return ValidationResult(passed=True, reason=f"judge unavailable: {exc}")

    async def validate(self, user_message: str, raw_text: str, sanitized_text: str, model: str) -> ValidationResult:
        leak_check = self.check_leak_safety(raw_text, sanitized_text)
        if not leak_check.passed:
            return leak_check
        return await self.check_quality(user_message, sanitized_text, model)
