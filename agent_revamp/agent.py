import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from agent_revamp.config import settings
from agent_revamp.core.mcp_tools import call_tool_safe
from agent_revamp.postprocess.leak_guard import sanitize_user_text
from agent_revamp.preprocess.process_class import ProcessClass, ToolScopeError, filter_tools, validate_tool_scope
from agent_revamp.postprocess.result_sanitizer import sanitize_tool_result
from agent_revamp.preprocess.schema_mapper import build_schema_prompt
from agent_revamp.preprocess.sql_translator import SQLTranslationError, translate_sql
from agent_revamp.preprocess.tool_sanitizer import inject_account_id, sanitize_tool_schema, translate_tool_args
from agent_revamp.core.state import SessionStore
from agent_revamp.core.vector import Embedder, SkillIndex, ToolIndex

_SQL_TOOLS = {"execute_query", "generate_report"}

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    """Lazy module-level singleton, exposed as a getter so it stays easy to mock in tests."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_key, base_url=settings.get_openai_base_url())
    return _openai_client


def _new_session_id() -> str:
    return secrets.token_hex(4)


_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools on the RealBooks penny-mcp server "
    "for querying and reporting on the platform's database."
)

# Map an internal tool name to a human-friendly progress caption (mirrors the original
# realbooks-agents harness for SSE `tools` events).
_TOOL_STATUS_LABELS = {
    "execute_query": "Looking that up",
    "generate_report": "Putting your report together",
    "get_account_profile": "Looking that up",
    "get_database_schema": "Looking that up",
    "search_market_data": "Searching the web",
    "fetch_page_content": "Reading a page",
}
_DEFAULT_STATUS_LABEL = "Working on it"

_MONEY_HINTS = (
    "income",
    "expense",
    "net",
    "total",
    "amount",
    "noi",
    "profit",
    "loss",
    "revenue",
    "cost",
    "balance",
    "rent",
    "payment",
    "budget",
    "spent",
)


def _status_label(tool_name: str) -> str:
    if tool_name in _TOOL_STATUS_LABELS:
        return _TOOL_STATUS_LABELS[tool_name]
    if tool_name.startswith("create_"):
        return "Saving your changes"
    if tool_name.startswith(("list_", "get_")):
        return "Looking that up"
    return _DEFAULT_STATUS_LABEL


def _fmt_money(value: float) -> str:
    return ("-$" if value < 0 else "$") + f"{abs(value):,.2f}"


def _format_key_figures(data_preview) -> str:
    """For a single-row totals/scalar report, surface the money figures by column name.
    Empty for multi-row or non-financial previews — the chart already shows those."""
    if not isinstance(data_preview, list) or len(data_preview) != 1 or not isinstance(data_preview[0], dict):
        return ""
    parts = [
        f"{key.replace('_', ' ').title()}: {_fmt_money(val)}"
        for key, val in data_preview[0].items()
        if isinstance(val, int | float)
        and not isinstance(val, bool)
        and any(hint in key.lower() for hint in _MONEY_HINTS)
    ]
    return " · ".join(parts)


def _report_summary_message(title: str, data_preview) -> str:
    """Confirmation shown with a generated report, built without a second LLM call."""
    base = f"Here is your **{title}**." if title else "Here is your report."
    figures = _format_key_figures(data_preview)
    return f"{base} {figures}" if figures else base


class Agent:
    def __init__(
        self,
        mcp_url: str | None = None,
        model: str | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
        max_iterations: int | None = None,
        session_id: str | None = None,
        state_dir: str | None = None,
        process_class: ProcessClass | None = None,
        account_id: int | None = None,
    ):
        self.mcp_url = mcp_url or settings.penny_mcp_url
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations or settings.max_tool_iterations
        # TODO(intent-classifier): process_class is fixed at construction time (config/caller-
        # supplied) rather than classified from the incoming message. The target architecture
        # has a dedicated Intent classifier stage upstream of Agent to make that call
        # dynamically per message — not built yet.
        self.process_class: ProcessClass = process_class or settings.process_class
        self.account_id: int | None = account_id
        self.system_prompt = system_prompt
        self.session_id = session_id or _new_session_id()
        self.store = SessionStore(state_dir or settings.state_dir)
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._deleted = False
        self._mcp: MCPClient | None = None
        self._openai_tools: list[dict] = []
        self._tool_index: ToolIndex | None = None
        self._skill_index: SkillIndex | None = None
        self.tokens_used = 0
        self.last_report_payload: dict | None = None

    async def __aenter__(self) -> "Agent":
        saved = self.store.load(self.session_id)
        if saved:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.messages.extend(m for m in saved.get("messages", []) if m.get("role") != "system")
            self._created_at = saved.get("created_at", self._created_at)

        self._mcp = MCPClient(self.mcp_url)
        await self._mcp.__aenter__()
        mcp_tools = await self._mcp.list_tools()
        # Hard-remove any tool this process class isn't allowed to use at all, before it's
        # ever sanitized, indexed into Qdrant, or shown to the model.
        mcp_tools = filter_tools(mcp_tools, self.process_class)
        self._openai_tools = [sanitize_tool_schema(tool) for tool in mcp_tools]
        self._tool_schemas = {t["function"]["name"]: t for t in self._openai_tools}

        # Dynamic skills & tools: index MCP tools and markdown skills into Qdrant so each
        # turn can retrieve only what is relevant. If Qdrant is down or indexing fails,
        # the agent degrades gracefully: all tools stay available, no skill context.
        try:
            embedder = Embedder(_get_openai())
            self._tool_index = ToolIndex(embedder=embedder)
            await self._tool_index.__aenter__()
            if not await self._tool_index.index_tools(mcp_tools):
                await self._tool_index.__aexit__(None, None, None)
                self._tool_index = None
            else:
                skill_texts = self._load_skills()
                if skill_texts:
                    self._skill_index = SkillIndex(embedder=embedder)
                    await self._skill_index.__aenter__()
                    if not await self._skill_index.index_skills(skill_texts):
                        await self._skill_index.__aexit__(None, None, None)
                        self._skill_index = None
        except Exception as exc:
            logger.warning("Qdrant init failed, continuing without vector index: %s", exc)
            self._tool_index = None
            self._skill_index = None

        # The model never sees penny-mcp's real db://schema resource (real table/column
        # names) — it gets the friendly-only catalog built from preprocess/schema_map.json
        # instead. sql_translator.translate_sql() converts the model's friendly-named SQL
        # to the real schema at dispatch time, in _dispatch below.
        self.messages[0]["content"] += "\n\n" + build_schema_prompt()

        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(*exc_info)
        if self._tool_index is not None:
            await self._tool_index.__aexit__(None, None, None)
        if self._skill_index is not None:
            await self._skill_index.__aexit__(None, None, None)
        self._persist()

    @staticmethod
    def _load_skills() -> list[tuple[str, str]]:
        """One skill per markdown file: returns (skill_name, text) pairs."""
        skills_dir = Path(settings.skills_dir)
        if not skills_dir.is_dir():
            return []
        skills = []
        for path in sorted(skills_dir.glob("*.md")):
            try:
                skills.append((path.stem, path.read_text(encoding="utf-8")))
            except OSError as exc:
                logger.warning("Could not read skill %s: %s", path, exc)
        return skills

    def _persist(self) -> None:
        if self._deleted:
            return
        self.store.save(self.session_id, self.messages, self.model, created_at=self._created_at)

    def delete_session(self) -> bool:
        """Delete this session's cached history from disk. The final save on
        __aexit__ is suppressed so the file is not recreated."""
        self._deleted = True
        return self.store.delete(self.session_id)

    def _history_for_llm(self, turn_start_idx: int, extra_system: list[str] | None = None) -> list[dict]:
        """System prompt + bounded persisted history + current-turn messages.

        Ported from realbooks-agents harness.py::_build_messages: history keeps the
        last MAX_HISTORY_MESSAGES messages, each truncated to MAX_MESSAGE_CHARS, so a
        long persisted session never blows up the context window. In-turn messages
        (user's latest question and the tool-call round-trips) are never trimmed.
        """
        messages: list[dict] = [self.messages[0]]
        for extra in extra_system or []:
            messages.append({"role": "system", "content": extra})
        for msg in self.messages[1:turn_start_idx][-settings.max_history_messages:]:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > settings.max_message_chars:
                msg = dict(msg)
                msg["content"] = content[: settings.max_message_chars]
            messages.append(msg)
        messages.extend(self.messages[turn_start_idx:])
        return messages

    async def _retrieve_tools(self, query: str) -> list[dict] | None:
        """Relevant tool schemas for the query, from Qdrant. None = fall back to all tools."""
        if self._tool_index is None:
            return None
        hits = await self._tool_index.search_tools(query)
        if not hits:
            return None
        return [self._tool_schemas[h["tool_name"]] for h in hits if h["tool_name"] in self._tool_schemas]

    async def _retrieve_skills(self, query: str) -> list[str]:
        if self._skill_index is None:
            return []
        chunks = await self._skill_index.search_skills(query)
        return chunks or []

    async def chat(self, user_message: str, event_sink: Callable[[dict], Awaitable[None]] | None = None) -> str:
        """Single conversational turn — runs the bounded tool-call loop internally and
        returns the model's final natural-language answer. History persists on
        self.messages across calls and to disk after each turn.

        With `event_sink` set (the API/SSE path), the turn runs in streaming mode and
        emits `tools`/`stage`/`token` events (sanitized per-chunk); without it, behavior
        is unchanged and the reply is returned whole.
        """
        turn_start_idx = len(self.messages)
        self.messages.append({"role": "user", "content": user_message})
        client = _get_openai()

        tools = await self._retrieve_tools(user_message)
        skills = await self._retrieve_skills(user_message)
        if tools is None:
            tools = self._openai_tools
        if skills:
            logger.info("Retrieved %d skill(s) for the turn", len(skills))

        # Second, independent check (filter_tools in __aenter__ is the first): assert every
        # tool about to be shown to the model this turn is within this process class's
        # allowlist. Guards specifically against Qdrant-based retrieval ever surfacing
        # something out of scope (e.g. a shared collection carrying entries from a session
        # run under a different process class) — see preprocess/process_class.py.
        try:
            validate_tool_scope([t["function"]["name"] for t in tools], self.process_class)
        except ToolScopeError as exc:
            logger.error("Tool-scope validation failed, refusing turn: %s", exc)
            self._persist()
            return "Internal error: this request was blocked by a tool-scope safety check."

        for _ in range(self.max_iterations):
            kwargs: dict = {
                "model": self.model,
                "messages": self._history_for_llm(
                    turn_start_idx,
                    extra_system=["--- Relevant skills for this request ---\n" + "\n\n".join(skills)] if skills else None,
                ),
            }
            if tools:
                kwargs.update(tools=tools, tool_choice="auto", parallel_tool_calls=True)

            if event_sink is not None:
                message, message_dict, usage = await self._stream_completion(kwargs, event_sink)
            else:
                response = await client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                message_dict = message.model_dump(exclude_none=True)
                usage = getattr(response.usage, "total_tokens", 0) or 0
            self.tokens_used += usage

            # Last line of defense: strip anything that slipped through the preprocess layer
            # (a hallucinated real name, a recited friendly label, pasted SQL) before this text
            # is shown to the user OR persisted — persisted history is replayed as context on
            # every future turn, so an unsanitized leak here would resurface indefinitely.
            if isinstance(message_dict.get("content"), str):
                message_dict["content"] = sanitize_user_text(message_dict["content"])
            self.messages.append(message_dict)

            if not message.tool_calls:
                self._persist()
                return message_dict.get("content") or ""

            if event_sink is not None:
                await event_sink(
                    {"type": "tools", "tools": list(dict.fromkeys(_status_label(tc.function.name) for tc in message.tool_calls))}
                )
                await event_sink({"type": "stage", "stage": "executing"})

            results = await asyncio.gather(*(self._dispatch(tc) for tc in message.tool_calls))
            for tool_call_id, content in results:
                self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

            # Skip the second LLM round-trip after a successful report — its only job is to
            # describe a chart the UI already renders (mirrors realbooks-agents harness).
            if self.last_report_payload and self.last_report_payload.get("token"):
                summary = sanitize_user_text(
                    _report_summary_message(
                        self.last_report_payload.get("title", ""),
                        self.last_report_payload.get("data_preview"),
                    )
                )
                self.messages.append({"role": "assistant", "content": summary})
                if event_sink is not None:
                    await event_sink({"type": "token", "content": summary})
                self._persist()
                return summary

        self._persist()
        return "Reached the maximum number of tool-call steps without a final answer."

    async def _stream_completion(self, openai_kwargs: dict, event_sink) -> tuple[object, dict, int]:
        """Stream a chat completion, emitting sanitized token events and reassembling
        tool-call deltas by index. Ported from realbooks-agents harness._stream_completion.
        Returns (message shim, message dict, token count) — shape-equivalent to the
        non-streamed path so the loop is unchanged.
        """
        from types import SimpleNamespace

        from agent_revamp.postprocess.stream_sanitizer import emit_sanitized

        kwargs = {**openai_kwargs, "stream": True, "stream_options": {"include_usage": True}}
        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        usage_tokens = 0
        emit_buf: dict = {}

        stream = await _get_openai().chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage:
                usage_tokens = chunk.usage.total_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                await emit_sanitized(event_sink, emit_buf, delta.content)
            for tcd in delta.tool_calls or []:
                slot = tool_acc.setdefault(tcd.index, {"id": None, "name": "", "arguments": ""})
                if tcd.id:
                    slot["id"] = tcd.id
                if tcd.function and tcd.function.name:
                    slot["name"] = tcd.function.name
                if tcd.function and tcd.function.arguments:
                    slot["arguments"] += tcd.function.arguments

        await emit_sanitized(event_sink, emit_buf, final=True)
        content = "".join(content_parts)
        tool_calls = [
            SimpleNamespace(
                id=s["id"],
                type="function",
                function=SimpleNamespace(name=s["name"], arguments=s["arguments"]),
            )
            for _, s in sorted(tool_acc.items())
        ]

        msg_dict: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        msg = SimpleNamespace(content=content or None, tool_calls=(tool_calls or None))
        return msg, msg_dict, usage_tokens

    async def _dispatch(self, tool_call) -> tuple[str, str]:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return tool_call.id, json.dumps({"error": f"invalid tool arguments: {exc}"})

        raw_args = args
        args = translate_tool_args(name, args)
        args = inject_account_id(name, args, self.account_id)
        if name in _SQL_TOOLS and "sql" in args:
            try:
                args["sql"] = translate_sql(args["sql"], account_id=args["account_id"])
            except SQLTranslationError as exc:
                return tool_call.id, json.dumps({"error": str(exc)})

        content = await call_tool_safe(self._mcp, name, args)
        content = sanitize_tool_result(content, name)
        if name == "generate_report":
            self.last_report_payload = _extract_report_payload(content, raw_args.get("title"))
        return tool_call.id, content
