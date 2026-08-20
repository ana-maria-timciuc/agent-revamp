import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from agent_revamp.config import settings
from agent_revamp.mcp_tools import call_tool_safe
from agent_revamp.preprocess.result_sanitizer import sanitize_tool_result
from agent_revamp.preprocess.schema_mapper import build_schema_prompt
from agent_revamp.preprocess.sql_translator import translate_sql
from agent_revamp.preprocess.tool_sanitizer import inject_account_id, sanitize_tool_schema, translate_tool_args
from agent_revamp.state import SessionStore
from agent_revamp.vector import Embedder, SkillIndex, ToolIndex

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


class Agent:
    def __init__(
        self,
        mcp_url: str | None = None,
        model: str | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
        max_iterations: int | None = None,
        session_id: str | None = None,
        state_dir: str | None = None,
    ):
        self.mcp_url = mcp_url or settings.penny_mcp_url
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations or settings.max_tool_iterations
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

    async def __aenter__(self) -> "Agent":
        saved = self.store.load(self.session_id)
        if saved:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.messages.extend(m for m in saved.get("messages", []) if m.get("role") != "system")
            self._created_at = saved.get("created_at", self._created_at)

        self._mcp = MCPClient(self.mcp_url)
        await self._mcp.__aenter__()
        mcp_tools = await self._mcp.list_tools()
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

    async def chat(self, user_message: str) -> str:
        """Single conversational turn — runs the bounded tool-call loop internally and
        returns the model's final natural-language answer. History persists on
        self.messages across calls and to disk after each turn."""
        turn_start_idx = len(self.messages)
        self.messages.append({"role": "user", "content": user_message})
        client = _get_openai()

        tools = await self._retrieve_tools(user_message)
        skills = await self._retrieve_skills(user_message)
        if tools is None:
            tools = self._openai_tools
        if skills:
            logger.info("Retrieved %d skill(s) for the turn", len(skills))

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

            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                self._persist()
                return message.content or ""

            results = await asyncio.gather(*(self._dispatch(tc) for tc in message.tool_calls))
            for tool_call_id, content in results:
                self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        self._persist()
        return "Reached the maximum number of tool-call steps without a final answer."

    async def _dispatch(self, tool_call) -> tuple[str, str]:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return tool_call.id, json.dumps({"error": f"invalid tool arguments: {exc}"})

        args = translate_tool_args(name, args)
        args = inject_account_id(name, args)
        if name in _SQL_TOOLS and "sql" in args:
            args["sql"] = translate_sql(args["sql"], account_id=args["account_id"])

        content = await call_tool_safe(self._mcp, name, args)
        content = sanitize_tool_result(content, name)
        return tool_call.id, content
