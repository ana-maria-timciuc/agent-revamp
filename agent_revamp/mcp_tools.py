"""MCP tool-schema conversion and result extraction — ported (trimmed) from realbooks-agents.

- extract_content(): app/agent/mcp_bridge.py::_extract_content, verbatim.
- tools_to_openai_schema(): the plain conversion shape used in
  app/agent/tool_schema_sanitizer.py::sanitize_tool_schema, WITHOUT the schema_map.json-driven
  friendly-rename layer (platform-specific, not applicable here — no schema_map.json exists
  in this project, so tool schemas pass through with their real MCP names/params as-is).
"""

import json
from typing import Any


def tools_to_openai_schema(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP Tool objects (from Client.list_tools()) to OpenAI function-tool dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


def extract_content(result: Any) -> str:
    """Unwrap a fastmcp call_tool() result into a plain string.

    Ported verbatim from realbooks-agents app/agent/mcp_bridge.py::_extract_content.
    """
    if result is None:
        return "{}"
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts) if parts else "{}"
    if hasattr(result, "content"):
        return extract_content(result.content)
    return json.dumps(result)


async def call_tool_safe(mcp_client: Any, name: str, arguments: dict[str, Any]) -> str:
    """Call an MCP tool, normalizing any failure into a JSON error string rather than
    raising — keeps one bad tool call from killing the whole agent loop iteration."""
    try:
        result = await mcp_client.call_tool(name, arguments)
        return extract_content(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
