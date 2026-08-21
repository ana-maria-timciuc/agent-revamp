"""Streaming leak guard: emits assistant text token-by-token through the safety net.

Ported from realbooks-agents/app/agent/harness.py's `_emit_sanitized` — the buffered
streaming wrapper leak_guard.py's docstring says must be ported before streaming is
added. Tokens arrive one fragment at a time, so a leak (a SQL snippet, a fenced block)
can span several tokens; the buffer only flushes on sentence/line boundaries so the
sanitizer gets whole sentences to match against. An unclosed ``` code fence is held
until it closes (or the buffer grows large), so a fenced block is redacted as a whole.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from agent_revamp.postprocess.leak_guard import sanitize_user_text

_EventSink = Callable[[dict], Awaitable[None]]

# Same flush thresholds as the original harness.
_MAX_BUFFER_CHARS = 2000
_FENCE_LIMIT_CHARS = 2000


async def emit_sanitized(
    emit: _EventSink,
    buf_state: dict,
    new_text: str = "",
    *,
    final: bool = False,
) -> None:
    """Buffer `new_text` and emit sanitized chunks once sentence/line boundaries arrive.

    `buf_state` is a caller-owned mutable dict ({}) shared across calls for one reply.
    Pass `final=True` on the last call to flush whatever remains.
    """
    buf_state["buf"] = buf_state.get("buf", "") + new_text
    buf = buf_state["buf"]

    if final:
        if buf:
            await emit({"type": "token", "content": sanitize_user_text(buf)})
        buf_state["buf"] = ""
        return

    # Hold inside an unclosed code fence unless the buffer is already large.
    if buf.count("```") % 2 == 1 and len(buf) < _FENCE_LIMIT_CHARS:
        return

    last = None
    for match in re.finditer(r"[.!?\n]", buf):
        last = match
    if last is not None:
        cut = last.end()
    elif len(buf) >= _MAX_BUFFER_CHARS:
        cut = len(buf)
    else:
        return

    head, tail = buf[:cut], buf[cut:]
    if head:
        await emit({"type": "token", "content": sanitize_user_text(head)})
    buf_state["buf"] = tail
