"""Request/response models for the /agent/* API.

Field-for-field mirror of realbooks-agents/app/api/chat.py so the consumer
contract (reporting-papi, webapp) stays unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str
    sql: str | None = None
    data_preview: list | None = None


class AgentChatRequest(BaseModel):
    prompt: str | None = None
    chat_history: list[ChatMessage] = []
    account_id: int = 0
    user_id: int = 0
    user_name: str = ""
    question: int | None = None
    params: str | None = None
    skip_account_id: bool = False
    is_reply: bool = False
    tz_offset_minutes: int | None = None
    permissions: list[str] = []
    role_key: str = ""


class AgentChatResponse(BaseModel):
    message: str = ""
    token: str | None = None
    data_preview: list | None = None
    question_id: int | None = None
    explanation: str | None = None
    sql: str | None = None
    tokens_used: int = 0
    needs_clarification: bool = False
    missing_fields: list | None = None
    intended_entity: str | None = None
    write_occurred: bool = False
    approval_required: bool = False
    approval_token: str | None = None
    tool: str | None = None
    human_readable: str | None = None
