import asyncio
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from agent_revamp.config import settings
from agent_revamp.core.mcp_tools import call_tool_safe
from agent_revamp.postprocess.leak_guard import redact_real_schema_leaks, sanitize_user_text
from agent_revamp.preprocess.process_class import ProcessClass, ToolScopeError, filter_tools, validate_tool_scope
from agent_revamp.postprocess.result_sanitizer import sanitize_tool_result
from agent_revamp.preprocess.schema_mapper import build_schema_prompt
from agent_revamp.preprocess.sql_translator import SQLTranslationError, translate_sql
from agent_revamp.preprocess.tool_sanitizer import inject_account_id, sanitize_tool_schema, translate_tool_args
from agent_revamp.core.state import SessionStore
from agent_revamp.preprocess.catalog import CatalogEntry, KIND_SKILL, KIND_TOOL, QdrantCatalog
from agent_revamp.preprocess.embeddings import OpenAIEmbeddingService
from agent_revamp.preprocess.intent import LLMIntentClassifier
from agent_revamp.preprocess.pipeline import ContextPackage, PreprocessPipeline
from agent_revamp.postprocess.validation import ResponseValidator

_SQL_TOOLS = {"execute_query", "generate_report"}

_REPORT_KEEP_KEYS = (
    "token",
    "question_id",
    "figure_json",
    "has_account_id_param",
    "status",
    "viz_type",
    "row_count",
    "data_preview",
)


def _extract_report_payload(content: str, title: str | None) -> dict:
    """Capture the user-facing fields of a sanitized generate_report result.

    Mirrors the original harness's report handling (harness.py::_REPORT_TOOLS branch):
    the metabase token/question_id/etc. are relayed to the caller as-is; the sanitizer
    already stripped everything else, so this is safe to surface.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    payload = {k: parsed.get(k) for k in _REPORT_KEEP_KEYS if parsed.get(k) is not None}
    if title:
        payload["title"] = title
    return payload

def _is_error_result(content: str) -> bool:
    """Best-effort check for call_tool_safe's `{"error": ...}` shape (timeout, exception,
    or an invalid-argument/SQL-translation rejection), used only to label a turn's tool
    trace ok/error for the state manager — never affects control flow."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


# Upper bound on per-session turn traces kept in the state file, mirroring the same
# "don't let a long-lived session grow unbounded" reasoning as MAX_HISTORY_MESSAGES for
# messages — a trace entry is small, but a session reused for months shouldn't grow forever.
_MAX_TRACKED_TURNS = 200

# Returned in place of a model-authored answer that failed postprocess Validation and
# exhausted its retry budget (agent.py::chat, non-streaming path) — never the flagged
# text itself, even sanitized.
_VALIDATION_FALLBACK_MESSAGE = "I couldn't put together a clean answer to that — could you try rephrasing your question?"

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
        # process_class is fixed at construction time (config/caller-supplied) rather than
        # classified from the incoming message — the intent classifier (preprocess/intent.py)
        # runs per-message via self._pipeline, but only informs the prompt (see chat()); it
        # doesn't drive which process_class/tool-allowlist this Agent instance runs as.
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
        self._pipeline: PreprocessPipeline | None = None
        self._validator: ResponseValidator | None = None
        self.tokens_used = 0
        self.last_report_payload: dict | None = None
        # Per-response trace for the state manager: time, tokens, tools, and the status
        # of every stage the response passed through (see _record_turn). Persisted
        # alongside self.messages, loaded back in __aenter__ for a resumed session.
        self.turns: list[dict] = []

    async def __aenter__(self) -> "Agent":
        saved = self.store.load(self.session_id)
        if saved:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.messages.extend(m for m in saved.get("messages", []) if m.get("role") != "system")
            self._created_at = saved.get("created_at", self._created_at)
            self.turns = list(saved.get("turns", []))

        self._mcp = MCPClient(self.mcp_url)
        await self._mcp.__aenter__()
        mcp_tools = await self._mcp.list_tools()
        # Hard-remove any tool this process class isn't allowed to use at all, before it's
        # ever sanitized, indexed into Qdrant, or shown to the model.
        mcp_tools = filter_tools(mcp_tools, self.process_class)
        self._openai_tools = [sanitize_tool_schema(tool) for tool in mcp_tools]
        self._tool_schemas = {t["function"]["name"]: t for t in self._openai_tools}

        # Dynamic skills & tools: index MCP tools and markdown skills into Qdrant so each
        # turn can retrieve only what is relevant (preprocess/pipeline.py, the diagram's
        # Preprocess box: intent classifier -> RAG -> reranker). If Qdrant is down or
        # indexing fails, the agent degrades gracefully: all tools stay available, no
        # skill context, no intent surfaced.
        try:
            catalog = QdrantCatalog(embedder=OpenAIEmbeddingService())
            tool_entries = [
                CatalogEntry(
                    id=t.name, kind=KIND_TOOL, name=t.name,
                    content=f"{t.name}\n{t.description or ''}", agent=self.process_class,
                )
                for t in mcp_tools
            ]
            if not await catalog.upsert(tool_entries):
                await catalog.close()
                self._pipeline = None
            else:
                skill_entries = [
                    CatalogEntry(id=name, kind=KIND_SKILL, name=name, content=text, agent=self.process_class)
                    for name, text in self._load_skills()
                ]
                if skill_entries and not await catalog.upsert(skill_entries):
                    logger.warning("Skill indexing failed; continuing with tool retrieval only")
                self._pipeline = PreprocessPipeline(
                    classifier=LLMIntentClassifier(client=_get_openai()), catalog=catalog
                )
        except Exception as exc:
            logger.warning("Qdrant init failed, continuing without vector index: %s", exc)
            self._pipeline = None

        # Postprocess "Validation" gate (diagram: Postprocess -> Validation -> Yes/No),
        # see postprocess/validation.py. Independent of the Qdrant pipeline above — only
        # needs an OpenAI client, which the rest of the agent already requires to function
        # at all — but guarded the same defensive way so a construction hiccup degrades
        # to "no gate" instead of blocking startup.
        try:
            self._validator = ResponseValidator(client=_get_openai())
        except Exception as exc:
            logger.warning("Response validator init failed, continuing without it: %s", exc)
            self._validator = None

        # The model never sees penny-mcp's real db://schema resource (real table/column
        # names) — it gets the friendly-only catalog built from preprocess/schema_map.json
        # instead. sql_translator.translate_sql() converts the model's friendly-named SQL
        # to the real schema at dispatch time, in _dispatch below.
        self.messages[0]["content"] += "\n\n" + build_schema_prompt()

        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(*exc_info)
        if self._pipeline is not None:
            await self._pipeline.close()
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
        self.store.save(
            self.session_id, self.messages, self.model, created_at=self._created_at, turns=self.turns
        )

    def _record_turn(
        self,
        started_at: datetime,
        clock_start: float,
        tokens_before: int,
        variables: dict,
        stages: dict[str, str],
        tools_trace: list[dict],
    ) -> None:
        """Append one entry to self.turns — the state manager's per-response trace:
        wall-clock timing, tokens spent this turn, the context variables that shaped it,
        the status of every stage the diagram's Preprocess/MCP/Postprocess boxes cover,
        and every tool call dispatched. Persisted by the next _persist() call."""
        self.turns.append(
            {
                "turn_index": len(self.turns),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.monotonic() - clock_start) * 1000, 1),
                "tokens_used": self.tokens_used - tokens_before,
                "tokens_total": self.tokens_used,
                "variables": variables,
                "stages": stages,
                "tools": tools_trace,
            }
        )
        if len(self.turns) > _MAX_TRACKED_TURNS:
            self.turns = self.turns[-_MAX_TRACKED_TURNS:]

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

    async def _run_preprocessing(self, user_message: str) -> ContextPackage | None:
        """Run the Preprocess box (intent classifier -> RAG -> reranker) once per turn.
        None = no pipeline available (Qdrant/indexing failed in __aenter__) -> caller
        falls back to the full tool set, no skills, no intent."""
        if self._pipeline is None:
            return None
        return await self._pipeline.run(user_message)

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

        turn_started_at = datetime.now(timezone.utc)
        turn_clock_start = time.monotonic()
        tokens_before = self.tokens_used
        stages: dict[str, str] = {"report_short_circuit": "no"}
        tools_trace: list[dict] = []
        validation_attempts = 0

        package = await self._run_preprocessing(user_message)
        if package is None:
            stages["preprocess"] = "skipped (no pipeline)"
        else:
            stages["preprocess"] = (
                f"ran (skills={len(package.skills)}, tools={len(package.tools)}, "
                f"skills_reranker={'passed' if package.skills_passed else 'failed'}, "
                f"tools_reranker={'passed' if package.tools_passed else 'failed'})"
            )

        # Three-way tool-candidate fallback: no pipeline, or the pipeline retrieved zero
        # tool candidates -> fall back to the full (process-class-filtered) tool set.
        # Retrieved candidates that don't survive the self._tool_schemas membership check
        # below yield an *empty* list, deliberately NOT the full set — a name outside the
        # process-class allowlist must never silently upgrade to "full access."
        if package is None or not package.tools:
            tools = self._openai_tools
        else:
            tools = [self._tool_schemas[e.name] for e in package.tools if e.name in self._tool_schemas]

        skills = [e.content for e in package.skills] if package else []
        if skills:
            logger.info("Retrieved %d skill(s) for the turn", len(skills))

        # Fold the classified intent + retrieved skills into the prompt (as separate
        # system-role messages, same mechanism _history_for_llm already uses for skills).
        # Intent is gated on is_confident (>= 0.6) so a low-confidence guess never
        # pollutes the prompt.
        extra_system: list[str] = []
        variables: dict = {
            "process_class": self.process_class,
            "account_id": self.account_id,
            "model": self.model,
            "tools_offered": len(tools),
        }
        if package is not None and package.intent.is_confident:
            extra_system.append(
                f"--- Detected intent: {package.intent.intent} (confidence={package.intent.confidence:.2f}) ---"
            )
            variables["intent"] = package.intent.intent
            variables["intent_confidence"] = round(package.intent.confidence, 2)
        if skills:
            extra_system.append("--- Relevant skills for this request ---\n" + "\n\n".join(skills))
            variables["skills_retrieved"] = len(skills)

        # Second, independent check (filter_tools in __aenter__ is the first): assert every
        # tool about to be shown to the model this turn is within this process class's
        # allowlist. Guards specifically against Qdrant-based retrieval ever surfacing
        # something out of scope (e.g. a shared collection carrying entries from a session
        # run under a different process class) — see preprocess/process_class.py.
        try:
            validate_tool_scope([t["function"]["name"] for t in tools], self.process_class)
            stages["tool_scope_validation"] = "passed"
        except ToolScopeError as exc:
            logger.error("Tool-scope validation failed, refusing turn: %s", exc)
            stages["tool_scope_validation"] = f"blocked: {exc}"
            self._record_turn(turn_started_at, turn_clock_start, tokens_before, variables, stages, tools_trace)
            self._persist()
            return "Internal error: this request was blocked by a tool-scope safety check."

        for _ in range(self.max_iterations):
            kwargs: dict = {
                "model": self.model,
                "messages": self._history_for_llm(turn_start_idx, extra_system=extra_system or None),
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
            raw_content = message_dict.get("content") if isinstance(message_dict.get("content"), str) else None
            if raw_content is not None:
                message_dict["content"] = sanitize_user_text(raw_content)
                stages["postprocess"] = "response sanitized (sanitize_user_text)"
            self.messages.append(message_dict)

            if not message.tool_calls:
                stages.setdefault("tool_dispatch", "skipped (no tool calls)")
                sanitized = message_dict.get("content") or ""

                # Postprocess "Validation" gate (diagram: Postprocess -> Validation ->
                # Yes: Answer / No: loop back to Agent). Only applies to a model-authored
                # final answer — the report short-circuit and max-iterations fallback
                # below aren't model-authored free text, so they mark "n/a" instead.
                if self._validator is None or raw_content is None:
                    stages["validation"] = "skipped (no validator)"
                elif event_sink is not None:
                    # Streaming: tokens were already emitted live to the client as they
                    # were generated (see _stream_completion) — there is nothing left to
                    # retract, so we still record the verdict for the state manager's
                    # trace but never retry.
                    result = await self._validator.validate(user_message, raw_content, sanitized, self.model)
                    stages["validation"] = (
                        "passed" if result.passed else f"failed: {result.reason} (not retried — streaming)"
                    )
                else:
                    result = await self._validator.validate(user_message, raw_content, sanitized, self.model)
                    if result.passed:
                        stages["validation"] = "passed"
                    elif validation_attempts < settings.max_validation_retries:
                        validation_attempts += 1
                        stages["validation"] = f"failed: {result.reason} (retry {validation_attempts})"
                        self.messages.pop()  # drop the rejected answer — never replay it as context
                        self.messages.append(
                            {
                                "role": "system",
                                "content": f"Your previous answer was rejected by validation "
                                f"({result.reason}). Answer again.",
                            }
                        )
                        continue
                    else:
                        stages["validation"] = f"failed: {result.reason} — retries exhausted, fallback returned"
                        self.messages.pop()
                        self.messages.append({"role": "assistant", "content": _VALIDATION_FALLBACK_MESSAGE})
                        self._record_turn(
                            turn_started_at, turn_clock_start, tokens_before, variables, stages, tools_trace
                        )
                        self._persist()
                        return _VALIDATION_FALLBACK_MESSAGE

                self._record_turn(turn_started_at, turn_clock_start, tokens_before, variables, stages, tools_trace)
                self._persist()
                return sanitized

            if event_sink is not None:
                await event_sink(
                    {"type": "tools", "tools": list(dict.fromkeys(_status_label(tc.function.name) for tc in message.tool_calls))}
                )
                await event_sink({"type": "stage", "stage": "executing"})

            results = await asyncio.gather(*(self._dispatch(tc) for tc in message.tool_calls))
            for tc, (tool_call_id, content, status) in zip(message.tool_calls, results, strict=True):
                tools_trace.append({"name": tc.function.name, "status": status})
                self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
            error_count = sum(1 for t in tools_trace if t["status"] == "error")
            stages["tool_dispatch"] = f"ok ({len(tools_trace)} call(s))" if not error_count else (
                f"{error_count}/{len(tools_trace)} call(s) errored"
            )
            stages.setdefault("postprocess", "tool results sanitized (result_sanitizer + leak_guard)")

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
                stages["report_short_circuit"] = "yes"
                stages["validation"] = "n/a (report short-circuit, not model-authored)"
                self._record_turn(turn_started_at, turn_clock_start, tokens_before, variables, stages, tools_trace)
                self._persist()
                return summary

        stages["tool_dispatch"] = stages.get("tool_dispatch", "n/a") + " — max iterations reached"
        stages["validation"] = "n/a (max iterations exhausted, not model-authored)"
        self._record_turn(turn_started_at, turn_clock_start, tokens_before, variables, stages, tools_trace)
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

    async def _dispatch(self, tool_call) -> tuple[str, str, str]:
        """Returns (tool_call_id, content, status) — status is "ok"/"error", used only
        by chat() to build this turn's state-manager tool trace, never for control flow."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            content = json.dumps({"error": f"invalid tool arguments: {exc}"})
            return tool_call.id, content, "error"

        raw_args = args
        args = translate_tool_args(name, args)
        args = inject_account_id(name, args, self.account_id)
        if name in _SQL_TOOLS and "sql" in args:
            try:
                args["sql"] = translate_sql(args["sql"], account_id=args["account_id"])
            except SQLTranslationError as exc:
                content = json.dumps({"error": str(exc)})
                return tool_call.id, content, "error"

        content = await call_tool_safe(self._mcp, name, args)
        content = sanitize_tool_result(content, name)
        # Defense-in-depth: sanitize_tool_result's per-tool-type allow-lists don't cover
        # every shape a result can take (e.g. a raw DB error surfacing from a "list"-type
        # tool like execute_query bypasses its allow-list entirely — see leak_guard.py's
        # module docstring). This is the same universal method that protects the
        # assistant's own replies, applied here before the content ever enters the
        # model's context or gets persisted to session history.
        content = redact_real_schema_leaks(content)
        if name == "generate_report":
            self.last_report_payload = _extract_report_payload(content, raw_args.get("title"))
        return tool_call.id, content, ("error" if _is_error_result(content) else "ok")
