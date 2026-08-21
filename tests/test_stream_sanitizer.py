"""Written from scratch -- realbooks-agents/tests/test_harness_streaming.py exercises
harness._stream_completion (OpenAI delta reassembly), not harness._emit_sanitized (the
function stream_sanitizer.py actually ports), and only ever feeds it plain safe text
where sanitization is a no-op. So there is no existing test of the buffering/leak-guard
behavior itself anywhere in realbooks-agents' suite; these cases were derived by reading
agent_revamp/postprocess/stream_sanitizer.py directly and confirmed by running it.

Covers: buffering with no sentence boundary, a sentence-boundary flush, an unclosed code
fence holding past a boundary character inside it, a schema leak arriving split across
two `new_text` calls (must be sanitized once assembled, not fragment-by-fragment), and
final=True on an empty buffer.
"""

import asyncio

from agent_revamp.postprocess.stream_sanitizer import emit_sanitized


def _run(coro):
    return asyncio.run(coro)


class _Collector:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)


def test_no_boundary_holds_until_final():
    emit = _Collector()
    buf_state: dict = {}
    _run(emit_sanitized(emit, buf_state, "Hello world"))
    assert emit.events == []
    assert buf_state["buf"] == "Hello world"

    _run(emit_sanitized(emit, buf_state, final=True))
    assert emit.events == [{"type": "token", "content": "Hello world"}]
    assert buf_state["buf"] == ""


def test_sentence_boundary_flushes_up_to_and_including_it():
    emit = _Collector()
    buf_state: dict = {}
    _run(emit_sanitized(emit, buf_state, "Hello world. More text"))
    assert emit.events == [{"type": "token", "content": "Hello world."}]
    assert buf_state["buf"] == " More text"


def test_unclosed_code_fence_holds_past_an_internal_boundary_char():
    emit = _Collector()
    buf_state: dict = {}
    _run(emit_sanitized(emit, buf_state, "Here's code: ```python\ndef foo():\n    pass.\n"))
    assert emit.events == [], "an odd number of ``` markers must hold, even past '.'/'\\n' inside the fence"

    _run(emit_sanitized(emit, buf_state, "``` and done."))
    assert len(emit.events) == 1
    content = emit.events[0]["content"]
    assert "[details omitted]" in content  # the whole fenced block is redacted as one unit
    assert "and done." in content
    assert buf_state["buf"] == ""


def test_leak_split_across_two_calls_is_sanitized_once_assembled():
    """The dangerous substring ("Transactions.PropertyId") never exists in either
    individual `new_text` fragment -- only once the buffer assembles them. Confirms
    sanitization runs on the whole assembled chunk, not fragment-by-fragment (which
    would miss it)."""
    emit = _Collector()
    buf_state: dict = {}
    _run(emit_sanitized(emit, buf_state, "Ref: Transactions"))
    assert emit.events == []

    _run(emit_sanitized(emit, buf_state, ".PropertyId is a real column."))
    assert len(emit.events) == 1
    content = emit.events[0]["content"]
    assert "Transactions.PropertyId" not in content
    assert "[details omitted]" in content


def test_final_with_empty_buffer_emits_nothing():
    emit = _Collector()
    buf_state: dict = {}
    _run(emit_sanitized(emit, buf_state, final=True))
    assert emit.events == []


def test_buf_state_is_caller_owned_and_reusable_across_independent_streams():
    """Two independent buf_state dicts must never interfere with each other."""
    emit_a, emit_b = _Collector(), _Collector()
    buf_a: dict = {}
    buf_b: dict = {}
    _run(emit_sanitized(emit_a, buf_a, "no boundary here"))
    _run(emit_sanitized(emit_b, buf_b, "also no boundary"))
    assert buf_a["buf"] == "no boundary here"
    assert buf_b["buf"] == "also no boundary"
