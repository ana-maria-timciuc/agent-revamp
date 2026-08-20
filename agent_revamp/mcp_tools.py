"""MCP result extraction — ported (trimmed) from realbooks-agents.

- extract_content(): app/agent/mcp_bridge.py::_extract_content, verbatim.

Tool-schema sanitization now lives in agent_revamp/preprocess/tool_sanitizer.py, since the
model must only ever see friendly (non-real) tool schemas — see that module.
"""

import json
from typing import Any


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
