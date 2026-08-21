"""Unit tests for postprocess/validation.py's ResponseValidator: the diagram's postprocess
"Validation" gate (distinct from the tool-scope "Validate tools" gate covered by
test_process_class.py). No live OpenAI needed -- a fake client stands in for AsyncOpenAI.
"""

import asyncio
import json
from types import SimpleNamespace

from agent_revamp.postprocess.validation import ResponseValidator, ValidationResult


def _run(coro):
    return asyncio.run(coro)


def _fake_client(payload=None, raise_exc=None):
    async def create(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        message = SimpleNamespace(content=json.dumps(payload if payload is not None else {"passed": True, "reason": ""}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_check_leak_safety_passes_when_sanitizer_was_a_no_op():
    validator = ResponseValidator(client=_fake_client())
    result = validator.check_leak_safety("Here is your total: $500", "Here is your total: $500")
    assert result.passed


def test_check_leak_safety_fails_when_sanitizer_changed_the_text():
    validator = ResponseValidator(client=_fake_client())
    result = validator.check_leak_safety(
        "Unknown column 'flow_type' in table `transaction`", "Unknown column 'Type' in [details omitted]"
    )
    assert not result.passed
    assert "redact" in result.reason


def test_check_quality_fails_on_a_negative_judge_verdict():
    validator = ResponseValidator(client=_fake_client(payload={"passed": False, "reason": "does not answer the question"}))
    result = _run(validator.check_quality("what's my income?", "I like turtles.", "gpt-5"))
    assert not result.passed
    assert result.reason == "does not answer the question"


def test_check_quality_passes_on_a_positive_judge_verdict():
    validator = ResponseValidator(client=_fake_client(payload={"passed": True, "reason": "on topic"}))
    result = _run(validator.check_quality("what's my income?", "Your income was $5,000.", "gpt-5"))
    assert result.passed


def test_check_quality_fails_open_when_the_judge_call_raises():
    validator = ResponseValidator(client=_fake_client(raise_exc=RuntimeError("network down")))
    result = _run(validator.check_quality("what's my income?", "Your income was $5,000.", "gpt-5"))
    assert result.passed  # fail OPEN -- a judge outage must never block a turn
    assert "judge unavailable" in result.reason


def test_check_quality_fails_open_on_malformed_json():
    validator = ResponseValidator(client=_fake_client())

    async def create(**kwargs):
        message = SimpleNamespace(content="not json at all")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    validator._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = _run(validator.check_quality("q", "a", "gpt-5"))
    assert result.passed


def test_validate_short_circuits_before_the_judge_when_leak_safety_already_failed():
    """The cheap deterministic check runs first; if it already fails there is no reason
    to spend an extra LLM call confirming what the sanitizer already caught."""

    async def create(**kwargs):
        raise AssertionError("the quality judge must not be called when leak-safety already failed")

    validator = ResponseValidator(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    result = _run(
        validator.validate(
            "what happened?",
            "Unknown column 'flow_type' in table `transaction`",
            "Unknown column 'Type' in [details omitted]",
            "gpt-5",
        )
    )
    assert not result.passed


def test_validate_runs_the_judge_when_leak_safety_passed():
    validator = ResponseValidator(client=_fake_client(payload={"passed": False, "reason": "off topic"}))
    result = _run(validator.validate("what's my income?", "I like turtles.", "I like turtles.", "gpt-5"))
    assert not result.passed
    assert result.reason == "off topic"


def test_validation_result_is_a_plain_dataclass():
    r = ValidationResult(passed=True, reason="")
    assert r.passed is True
    assert r.reason == ""
