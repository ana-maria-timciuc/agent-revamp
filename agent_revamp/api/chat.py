"""/agent/* endpoints — seamless replacement for realbooks-agents.

The contract mirrors realbooks-agents/app/api/chat.py exactly: same paths, same
request/response models, same SSE event stream (stage/tools/token/result/error,
keep-alives). reporting-papi proxies every /agent/{path} to `agents_api`, so
pointing that URL at this service requires no consumer changes.

Only the penny agent is wired up here (agent-revamp has a single MCP server);
dollar-bill / uncle-sam slugs 404, and approvals are unsupported (penny's tools
are read-only, so approval_required is never emitted).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent_revamp.agent import Agent
from agent_revamp.api.auth import UserIdentity, get_verified_identity
from agent_revamp.api.models import AgentChatRequest, AgentChatResponse
from agent_revamp.api.sse import make_stream_response

logger = logging.getLogger(__name__)
router = APIRouter()

# Normalize URL slug to registry key (mirrors the original _SLUG_MAP; only penny wired).
_SLUG_MAP = {
    "penny": "penny",
    "dollar-bill": "dollar_bill",
    "dollar_bill": "dollar_bill",
    "uncle-sam": "uncle_sam",
    "uncle_sam": "uncle_sam",
}

_REGISTRY = {
    "penny": {
        "display_name": "Penny",
        "role": "AI Bookkeeper",
        "tagline": "Tracks every dollar, automatically.",
    }
}


def _resolve_agent_type(slug: str) -> str:
    resolved = _SLUG_MAP.get(slug.lower())
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    if resolved not in _REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    return resolved


class ApproveRequest(BaseModel):
    approval_token: str
    approved: bool


def _format_tz_offset(minutes: int | None) -> str:
    """Convert JS getTimezoneOffset() (negative-ahead) to SQL ±HH:MM string."""
    if minutes is None:
        return "+00:00"
    total = -minutes
    sign = "+" if total >= 0 else "-"
    h, m = divmod(abs(total), 60)
    return f"{sign}{h:02d}:{m:02d}"


def _build_messages(request: AgentChatRequest, system_prompt: str) -> list[dict]:
    """System prompt + account/date/timezone context + bounded chat history.

    Ported from realbooks-agents harness._build_messages (minus the current prompt,
    which Agent.chat() appends itself): history keeps the last 12 messages, each
    truncated to 3000 chars.
    """
    today = date.today().isoformat()
    tz_sql = _format_tz_offset(request.tz_offset_minutes)
    account_ctx = (
        f"\n\n---\n"
        f"**Today's date: {today}**\n"
        f"Always use {today} when the user says 'today', 'azi', 'astazi', 'now', or any present-day reference.\n"
        f"Never use a hardcoded or assumed date — always use {today} for current date.\n"
        f"**Session account_id: {request.account_id}**\n"
        f"Account scoping is automatic — you never write account_id filters yourself.\n"
        f"**User timezone: UTC{tz_sql}**\n"
        f"Date values are automatically converted to the session timezone. Reference the Date column directly "
        f"in your SQL — the translator handles the timezone conversion."
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt + account_ctx}]
    for msg in (request.chat_history or [])[-12:]:
        content = msg.content[:3000] if len(msg.content) > 3000 else msg.content
        messages.append({"role": msg.role, "content": content})
    return messages


def _effective_prompt(request: AgentChatRequest) -> str:
    user_content = request.prompt or ""
    if request.is_reply:
        user_content = f"[REPLY TO PREVIOUS MESSAGE] {user_content}"
    return user_content


def _result_from_agent(agent: Agent, reply: str) -> dict:
    result: dict = {
        "message": reply,
        "tokens_used": agent.tokens_used,
    }
    report = agent.last_report_payload or {}
    for key in ("token", "question_id", "data_preview", "has_account_id_param"):
        if report.get(key) is not None:
            result[key] = report[key]
    return result


async def _run_agent_request(request: AgentChatRequest, identity: UserIdentity, agent_type: str) -> Agent:
    account_id = identity.account_id
    if not account_id or account_id <= 0:
        raise HTTPException(status_code=403, detail="Missing account identity. Please log in again.")
    agent = Agent(account_id=account_id)
    await agent.__aenter__()
    try:
        agent.messages = _build_messages(request, agent.messages[0]["content"])
    except Exception:
        await agent.__aexit__(None, None, None)
        raise
    return agent


@router.post("/agent/chat", response_model=AgentChatResponse)
async def chat_legacy(request: AgentChatRequest, identity: UserIdentity = Depends(get_verified_identity)):
    """Legacy endpoint (backward compatible — routes to Penny)."""
    agent = await _run_agent_request(request, identity, "penny")
    try:
        reply = await agent.chat(_effective_prompt(request))
    finally:
        await agent.__aexit__(None, None, None)
    return AgentChatResponse(**_result_from_agent(agent, reply))


@router.post("/agent/chat/stream")
async def chat_stream_legacy(request: AgentChatRequest, identity: UserIdentity = Depends(get_verified_identity)):
    async def producer(queue: asyncio.Queue):
        await _stream_turn(request, identity, "penny", queue)

    return make_stream_response(producer)


@router.post("/agent/{agent_slug}/chat", response_model=AgentChatResponse)
async def chat(agent_slug: str, request: AgentChatRequest, identity: UserIdentity = Depends(get_verified_identity)):
    agent_type = _resolve_agent_type(agent_slug)
    agent = await _run_agent_request(request, identity, agent_type)
    try:
        reply = await agent.chat(_effective_prompt(request))
    finally:
        await agent.__aexit__(None, None, None)
    return AgentChatResponse(**_result_from_agent(agent, reply))


@router.post("/agent/{agent_slug}/chat/stream")
async def chat_stream(
    agent_slug: str, request: AgentChatRequest, identity: UserIdentity = Depends(get_verified_identity)
):
    agent_type = _resolve_agent_type(agent_slug)

    async def producer(queue: asyncio.Queue):
        await _stream_turn(request, identity, agent_type, queue)

    return make_stream_response(producer)


async def _stream_turn(
    request: AgentChatRequest, identity: UserIdentity, agent_type: str, queue: asyncio.Queue
) -> None:
    async def emit(event: dict):
        await queue.put(event)

    await emit({"type": "stage", "stage": "routing"})

    account_id = identity.account_id
    if not account_id or account_id <= 0:
        await emit({"type": "error", "message": "Missing account identity. Please log in again."})
        return

    agent = Agent(account_id=account_id)
    await agent.__aenter__()
    try:
        agent.messages = _build_messages(request, agent.messages[0]["content"])
        reply = await agent.chat(_effective_prompt(request), event_sink=emit)
    finally:
        await agent.__aexit__(None, None, None)

    await emit({"type": "result", "data": _result_from_agent(agent, reply)})


@router.post("/agent/{agent_slug}/approve")
async def approve_action(
    agent_slug: str,
    body: ApproveRequest,
    identity: UserIdentity = Depends(get_verified_identity),
):
    """Approvals are not supported: agent-revamp only wires penny, whose tools are
    read-only — approval_required is never emitted, so the webapp never calls this."""
    _resolve_agent_type(agent_slug)  # validate slug
    raise HTTPException(status_code=404, detail="This agent does not support approvals.")


@router.get("/agent/registry")
async def get_registry():
    return _REGISTRY
