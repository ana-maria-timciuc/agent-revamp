"""Unit test for agent_revamp/preprocess/pipeline.py::PreprocessPipeline -- confirms it
correctly assembles a ContextPackage from an injected classifier + catalog (no real
OpenAI/Qdrant calls), and that preprocess_text() normalizes whitespace before either is
called.
"""

import asyncio

from agent_revamp.preprocess.catalog import (
    KIND_SKILL,
    KIND_TOOL,
    CatalogEntry,
    RetrievalResult,
)
from agent_revamp.preprocess.intent import IntentClassifier, IntentResult
from agent_revamp.preprocess.pipeline import ContextPackage, PreprocessPipeline, preprocess_text


def _run(coro):
    return asyncio.run(coro)


class _FakeClassifier(IntentClassifier):
    async def classify(self, text):
        return IntentResult(intent="reporting", confidence=0.9, raw_label="{}")


class _FakeCatalog:
    async def search(self, query, top_k=None):
        skill = CatalogEntry(id="s1", kind=KIND_SKILL, name="reporting", content="skill text", agent="penny")
        tool = CatalogEntry(id="generate_report", kind=KIND_TOOL, name="generate_report", content="tool text", agent="penny")
        return RetrievalResult(skills=[(skill, 0.9)], tools=[(tool, 0.8)], skills_passed=True, tools_passed=True)

    async def close(self):
        pass


def test_preprocess_text_collapses_whitespace():
    assert preprocess_text("  what's   my rental income   ") == "what's my rental income"


def test_pipeline_assembles_a_context_package_from_classifier_and_catalog():
    pipeline = PreprocessPipeline(classifier=_FakeClassifier(), catalog=_FakeCatalog())
    package = _run(pipeline.run("  what's   my rental income   "))

    assert isinstance(package, ContextPackage)
    assert package.raw_query == "what's my rental income"
    assert package.intent.intent == "reporting"
    assert package.intent.is_confident
    assert [s.name for s in package.skills] == ["reporting"]
    assert [t.name for t in package.tools] == ["generate_report"]
    assert package.skills_passed is True
    assert package.tools_passed is True
    _run(pipeline.close())


def test_low_confidence_intent_result_is_still_carried_through():
    """The is_confident gate is Agent's concern (whether to surface it in the prompt),
    not the pipeline's -- run() must always report the classifier's real result."""

    class _UnsureClassifier(IntentClassifier):
        async def classify(self, text):
            return IntentResult(intent="off_domain", confidence=0.2, raw_label="{}")

    pipeline = PreprocessPipeline(classifier=_UnsureClassifier(), catalog=_FakeCatalog())
    package = _run(pipeline.run("hello"))
    assert package.intent.confidence == 0.2
    assert package.intent.is_confident is False
