import asyncio
import json

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from agent_revamp.config import settings
from agent_revamp.mcp_tools import call_tool_safe, tools_to_openai_schema

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    """Lazy module-level singleton, exposed as a getter so it stays easy to mock in tests."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_key, base_url=settings.get_openai_base_url())
    return _openai_client


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
    ):
        self.mcp_url = mcp_url or settings.penny_mcp_url
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations or settings.max_tool_iterations
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._mcp: MCPClient | None = None
        self._openai_tools: list[dict] = []

    async def __aenter__(self) -> "Agent":
        self._mcp = MCPClient(self.mcp_url)
        await self._mcp.__aenter__()
        mcp_tools = await self._mcp.list_tools()
        self._openai_tools = tools_to_openai_schema(mcp_tools)

        # penny-mcp's own instructions require reading the db://schema resource before
        # writing any SQL, but chat-completions tool-calling never exposes MCP *resources*
        # to the model (only tools) — so it has to be fetched here and folded into the
        # system prompt, or the model guesses table/column names blind and every
        # execute_query/generate_report call fails validation.
        try:
            schema_contents = await self._mcp.read_resource("db://schema")
            schema_text = "\n".join(c.text for c in schema_contents if hasattr(c, "text"))
            if schema_text:
                self.messages[0]["content"] += "\n\n--- Database schema and SQL rules ---\n" + schema_text
        except Exception:
            pass

        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(*exc_info)

    async def chat(self, user_message: str) -> str:
        """Single conversational turn — runs the bounded tool-call loop internally and
        returns the model's final natural-language answer. History persists on
        self.messages across calls."""
        self.messages.append({"role": "user", "content": user_message})
        client = _get_openai()

        for _ in range(self.max_iterations):
            kwargs: dict = {"model": self.model, "messages": self.messages}
            if self._openai_tools:
                kwargs.update(tools=self._openai_tools, tool_choice="auto", parallel_tool_calls=True)

            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            results = await asyncio.gather(*(self._dispatch(tc) for tc in message.tool_calls))
            for tool_call_id, content in results:
                self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        return "Reached the maximum number of tool-call steps without a final answer."

    async def _dispatch(self, tool_call) -> tuple[str, str]:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return tool_call.id, json.dumps({"error": f"invalid tool arguments: {exc}"})
        content = await call_tool_safe(self._mcp, tool_call.function.name, args)
        return tool_call.id, content
