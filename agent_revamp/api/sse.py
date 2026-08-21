"""SSE streaming response helper — ported from realbooks-agents/app/api/chat.py.

Runs the agent producer in the background, relays events to the client with 15s
keep-alives so long-running turns never look dead to the consumer proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 15


def make_stream_response(producer_coro) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        try:
            await producer_coro(queue)
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            await queue.put({"type": "error", "message": "Something went wrong. Please try again."})
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(producer())
        getter: asyncio.Task | None = None
        try:
            while True:
                if getter is None:
                    getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({getter}, timeout=_HEARTBEAT_SECONDS)
                if getter not in done:
                    yield ": keep-alive\n\n"
                    continue
                event = getter.result()
                getter = None
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            if getter is not None:
                getter.cancel()
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
