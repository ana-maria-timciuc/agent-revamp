"""Written from scratch. realbooks-agents has no test for mcp_bridge.py::_extract_content
(the function core/mcp_tools.py::extract_content ports "verbatim" per its own docstring)
-- confirmed via a repo-wide grep for the symbol, zero hits. Cases below were derived by
reading core/mcp_tools.py directly.
"""

import asyncio
import json
from types import SimpleNamespace

from agent_revamp.core.mcp_tools import call_tool_safe, extract_content


def test_extract_content_none_returns_empty_json_object():
    assert extract_content(None) == "{}"


def test_extract_content_string_passthrough():
    assert extract_content("already a string") == "already a string"


def test_extract_content_list_of_text_bearing_objects_joined_by_newline():
    items = [SimpleNamespace(text="first"), SimpleNamespace(text="second")]
    assert extract_content(items) == "first\nsecond"


def test_extract_content_list_of_dicts_with_text_key():
    items = [{"text": "one"}, {"text": "two"}]
    assert extract_content(items) == "one\ntwo"


def test_extract_content_list_with_no_text_bearing_items_falls_back_to_empty_object():
    assert extract_content([{"other": "field"}, 123]) == "{}"


def test_extract_content_recurses_into_a_content_attribute():
    wrapper = SimpleNamespace(content=[SimpleNamespace(text="nested")])
    assert extract_content(wrapper) == "nested"


def test_extract_content_falls_back_to_json_dumps_for_anything_else():
    assert extract_content({"a": 1}) == json.dumps({"a": 1})


def _run(coro):
    return asyncio.run(coro)


def test_call_tool_safe_returns_extracted_content_on_success():
    class FakeMCP:
        async def call_tool(self, name, arguments):
            assert name == "execute_query"
            assert arguments == {"sql": "SELECT 1"}
            return [SimpleNamespace(text='{"rows": []}')]

    out = _run(call_tool_safe(FakeMCP(), "execute_query", {"sql": "SELECT 1"}))
    assert out == '{"rows": []}'


def test_call_tool_safe_never_raises_on_a_hung_call_returns_timeout_error_json(monkeypatch):
    from agent_revamp.config import settings

    monkeypatch.setattr(settings, "tool_call_timeout_seconds", 0.05)

    class HangingMCP:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(1)
            return "never gets here"

    out = _run(call_tool_safe(HangingMCP(), "execute_query", {}))
    parsed = json.loads(out)
    assert "error" in parsed
    assert "timed out" in parsed["error"]
    assert "execute_query" in parsed["error"]


def test_call_tool_safe_never_raises_on_any_other_exception():
    class BrokenMCP:
        async def call_tool(self, name, arguments):
            raise ConnectionError("mcp server unreachable")

    out = _run(call_tool_safe(BrokenMCP(), "execute_query", {}))
    parsed = json.loads(out)
    assert parsed == {"error": "mcp server unreachable"}
