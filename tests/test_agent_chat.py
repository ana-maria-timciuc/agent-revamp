"""Unit tests for Agent.chat()'s preprocessing integration: the three-way tool-candidate
fallback and intent/skills prompt injection added when agent.py was rewired onto
PreprocessPipeline. No live MCP/Qdrant/OpenAI needed -- the OpenAI client is monkeypatched
at the module-level singleton agent.py uses (_get_openai()/_openai_client).
"""

import asyncio
import json
import tempfile
from types import SimpleNamespace

import agent_revamp.agent as agent_module
from agent_revamp.agent import Agent
from agent_revamp.config import settings
from agent_revamp.preprocess.catalog import CatalogEntry, KIND_SKILL, KIND_TOOL
from agent_revamp.preprocess.intent import IntentResult
from agent_revamp.preprocess.pipeline import ContextPackage
from agent_revamp.postprocess.validation import ValidationResult


def _run(coro):
    return asyncio.run(coro)


def _make_agent():
    # A dedicated temp state_dir per agent — otherwise every chat() call here would
    # persist a real session file into this project's actual state/ directory.
    agent = Agent(process_class="penny", state_dir=tempfile.mkdtemp(prefix="agent_revamp_test_state_"))
    agent._tool_schemas = {
        "execute_query": {"type": "function", "function": {"name": "execute_query", "parameters": {}}},
        "generate_report": {"type": "function", "function": {"name": "generate_report", "parameters": {}}},
    }
    agent._openai_tools = list(agent._tool_schemas.values())
    return agent


def _fake_openai_client(captured_kwargs):
    async def create(**kwargs):
        captured_kwargs.append(kwargs)
        message = SimpleNamespace(content="final answer", tool_calls=None)
        message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": "final answer"}
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=10))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _install_fake_openai(monkeypatch, captured_kwargs):
    monkeypatch.setattr(agent_module, "_openai_client", _fake_openai_client(captured_kwargs))


class _FakePipeline:
    def __init__(self, package_factory):
        self._package_factory = package_factory

    async def run(self, msg):
        return self._package_factory(msg)


class _FakeValidator:
    """Queue of canned ValidationResults, one per validate() call, for exercising
    chat()'s postprocess Validation loop-back without a real judge LLM call."""

    def __init__(self, results):
        self._results = list(results)
        self.call_count = 0

    async def validate(self, user_message, raw_text, sanitized_text, model):
        self.call_count += 1
        return self._results.pop(0)


def _fake_streaming_openai_client(call_log, content="final answer"):
    async def create(**kwargs):
        call_log.append(kwargs)

        async def gen():
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))])
            yield SimpleNamespace(usage=SimpleNamespace(total_tokens=10), choices=[])

        return gen()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_no_pipeline_falls_back_to_the_full_tool_set(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    captured = []
    _install_fake_openai(monkeypatch, captured)

    reply = _run(agent.chat("hello"))

    assert reply == "final answer"
    tool_names = {t["function"]["name"] for t in captured[0]["tools"]}
    assert tool_names == {"execute_query", "generate_report"}
    messages = captured[0]["messages"]
    assert not any("Detected intent" in m.get("content", "") for m in messages if m["role"] == "system")


def test_zero_retrieved_tool_candidates_falls_back_to_full_set(monkeypatch):
    agent = _make_agent()
    agent._pipeline = _FakePipeline(
        lambda msg: ContextPackage(
            intent=IntentResult(intent="lookup", confidence=0.9, raw_label="{}"), skills=[], tools=[], raw_query=msg
        )
    )
    captured = []
    _install_fake_openai(monkeypatch, captured)

    _run(agent.chat("hello"))

    tool_names = {t["function"]["name"] for t in captured[0]["tools"]}
    assert tool_names == {"execute_query", "generate_report"}


def test_out_of_scope_candidate_yields_empty_list_never_full_access(monkeypatch):
    """The security-critical case: a retrieved tool name that isn't in this process
    class's allowlist (e.g. a stale hit from another process class sharing the Qdrant
    collection) must produce an EMPTY tool list, never silently upgrade to full access."""
    agent = _make_agent()

    def make_package(msg):
        bogus = CatalogEntry(id="stage_rows", kind=KIND_TOOL, name="stage_rows", content="x", agent="transaction_saving")
        return ContextPackage(
            intent=IntentResult(intent="lookup", confidence=0.9, raw_label="{}"), skills=[], tools=[bogus], raw_query=msg
        )

    agent._pipeline = _FakePipeline(make_package)
    captured = []
    _install_fake_openai(monkeypatch, captured)

    _run(agent.chat("hello"))

    assert "tools" not in captured[0], "an empty tool list must omit the tools= kwarg entirely (matches `if tools:`)"


def test_intent_and_skills_are_surfaced_as_system_messages(monkeypatch):
    agent = _make_agent()

    def make_package(msg):
        skill = CatalogEntry(id="reporting", kind=KIND_SKILL, name="reporting", content="Prefer generate_report for charts.", agent="penny")
        return ContextPackage(
            intent=IntentResult(intent="reporting", confidence=0.87, raw_label="{}"), skills=[skill], tools=[], raw_query=msg
        )

    agent._pipeline = _FakePipeline(make_package)
    captured = []
    _install_fake_openai(monkeypatch, captured)

    _run(agent.chat("show me my income report"))

    messages = captured[0]["messages"]
    intent_msgs = [m for m in messages if m["role"] == "system" and "Detected intent: reporting" in m["content"]]
    skill_msgs = [m for m in messages if m["role"] == "system" and "Prefer generate_report" in m["content"]]
    assert len(intent_msgs) == 1
    assert "confidence=0.87" in intent_msgs[0]["content"]
    assert len(skill_msgs) == 1


def test_dispatch_redacts_a_raw_db_error_before_it_enters_the_model_or_persisted_history():
    """Regression test for the confirmed leak: sanitize_tool_result's allow-list for
    "list"-type tools (execute_query) doesn't touch an unexpected top-level key like
    "error", so a raw DB/driver error carrying real schema identifiers used to pass
    through untouched. _dispatch() now also runs redact_real_schema_leaks() on every
    tool result, regardless of tool type."""
    agent = _make_agent()

    class _FakeMCP:
        async def call_tool(self, name, arguments):
            assert arguments["sql"] == "SELECT 1"
            return json.dumps(
                {"error": "Unknown column 'flow_type' in 'field list' on table `transaction`.account_id=45"}
            )

    agent._mcp = _FakeMCP()
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="execute_query", arguments=json.dumps({"sql": "SELECT 1"})),
    )

    _, content, status = _run(agent._dispatch(tool_call))

    assert "flow_type" not in content
    assert "account_id" not in content
    assert status == "error"
    parsed = json.loads(content)
    assert "Type" in parsed["error"]  # substituted with the friendly name, not just blanked


def test_chat_records_a_turn_trace_with_timing_tokens_and_stage_status(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    captured = []
    _install_fake_openai(monkeypatch, captured)

    _run(agent.chat("hello"))

    assert len(agent.turns) == 1
    turn = agent.turns[0]
    assert turn["turn_index"] == 0
    assert turn["tokens_used"] == 10
    assert turn["tokens_total"] == 10
    assert turn["duration_ms"] >= 0
    assert turn["variables"]["process_class"] == "penny"
    assert turn["variables"]["tools_offered"] == 2
    assert turn["stages"]["preprocess"] == "skipped (no pipeline)"
    assert turn["stages"]["tool_scope_validation"] == "passed"
    assert turn["stages"]["tool_dispatch"] == "skipped (no tool calls)"
    assert turn["tools"] == []

    # Persisted alongside messages, and reloaded on session resume.
    reloaded = agent.store.load(agent.session_id)
    assert len(reloaded["turns"]) == 1
    assert reloaded["turns"][0]["tokens_used"] == 10


def test_chat_traces_a_failed_tool_call_as_error_status(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None

    call_count = {"n": 0}

    async def create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            tool_call = SimpleNamespace(
                id="call_1", function=SimpleNamespace(name="execute_query", arguments="not json")
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            message.model_dump = lambda exclude_none=True: {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "execute_query", "arguments": "not json"}}],
            }
        else:
            message = SimpleNamespace(content="final answer", tool_calls=None)
            message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": "final answer"}
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=5))

    monkeypatch.setattr(
        agent_module, "_openai_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )

    _run(agent.chat("hello"))

    turn = agent.turns[0]
    assert turn["tools"] == [{"name": "execute_query", "status": "error"}]
    assert turn["stages"]["tool_dispatch"] == "1/1 call(s) errored"
    assert turn["tokens_used"] == 10  # two round-trips, 5 tokens each


def test_low_confidence_intent_is_not_surfaced_in_the_prompt(monkeypatch):
    agent = _make_agent()
    agent._pipeline = _FakePipeline(
        lambda msg: ContextPackage(
            intent=IntentResult(intent="off_domain", confidence=0.3, raw_label="{}"), skills=[], tools=[], raw_query=msg
        )
    )
    captured = []
    _install_fake_openai(monkeypatch, captured)

    _run(agent.chat("hello"))

    messages = captured[0]["messages"]
    assert not any("Detected intent" in m.get("content", "") for m in messages if m["role"] == "system")


def test_validation_failure_loops_back_to_the_agent_then_succeeds(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    agent._validator = _FakeValidator(
        [ValidationResult(passed=False, reason="leaked schema"), ValidationResult(passed=True, reason="")]
    )
    captured = []
    _install_fake_openai(monkeypatch, captured)

    reply = _run(agent.chat("hello"))

    assert reply == "final answer"
    assert len(captured) == 2  # rejected once, retried, then accepted
    assert agent._validator.call_count == 2
    assert agent.turns[-1]["stages"]["validation"] == "passed"

    # The rejected first attempt must not survive into persisted/replayed history.
    assistant_messages = [m for m in agent.messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "final answer"
    rejection_notes = [m for m in agent.messages if m["role"] == "system" and "rejected by validation" in m.get("content", "")]
    assert len(rejection_notes) == 1


def test_validation_failure_exhausts_retries_and_returns_fallback(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    total_attempts = settings.max_validation_retries + 1
    agent._validator = _FakeValidator(
        [ValidationResult(passed=False, reason="leaked schema") for _ in range(total_attempts)]
    )
    captured = []
    _install_fake_openai(monkeypatch, captured)

    reply = _run(agent.chat("hello"))

    assert reply == agent_module._VALIDATION_FALLBACK_MESSAGE
    assert len(captured) == total_attempts
    assert agent._validator.call_count == total_attempts
    assert "retries exhausted" in agent.turns[-1]["stages"]["validation"]
    # The fallback is what's actually persisted, never the flagged text.
    assistant_messages = [m for m in agent.messages if m["role"] == "assistant"]
    assert assistant_messages[-1]["content"] == agent_module._VALIDATION_FALLBACK_MESSAGE


def test_streaming_path_records_a_validation_failure_but_never_retries(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    agent._validator = _FakeValidator([ValidationResult(passed=False, reason="off topic")])
    call_log = []
    monkeypatch.setattr(agent_module, "_openai_client", _fake_streaming_openai_client(call_log))

    events = []

    async def sink(event):
        events.append(event)

    reply = _run(agent.chat("hello", event_sink=sink))

    assert reply == "final answer"  # streamed tokens already sent -- never swapped for the fallback
    assert len(call_log) == 1  # no retry: nothing left to retract once tokens are out
    assert agent._validator.call_count == 1
    stage = agent.turns[-1]["stages"]["validation"]
    assert stage.startswith("failed:")
    assert "streaming" in stage


def test_validation_is_skipped_when_no_validator_is_configured(monkeypatch):
    agent = _make_agent()
    agent._pipeline = None
    agent._validator = None
    captured = []
    _install_fake_openai(monkeypatch, captured)

    reply = _run(agent.chat("hello"))

    assert reply == "final answer"
    assert len(captured) == 1
    assert agent.turns[-1]["stages"]["validation"] == "skipped (no validator)"
