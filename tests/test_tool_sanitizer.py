"""Ported from realbooks-agents/tests/test_tool_schema_sanitizer.py, adapted heavily:
agent-revamp's schema_map.json has NO "tool_arg_maps" key at all (penny-mcp's only args
-- account_id/sql/title/viz_type -- are already business-level names, per
tool_sanitizer.py's own docstring), so there is no parameter-renaming behavior to test
here the way the original repo's create_asset/create_transaction/... cases did. What
agent-revamp's sanitize_tool_schema/translate_tool_args/inject_account_id actually do,
for its only two allowed tools (execute_query, generate_report), is: drop the hidden
account_id parameter, and rewrite the tool description to remove real schema mentions.

Ground-truth outputs below were captured by running these functions directly against
this repo's actual schema_map.json and process_class.py registry.
"""

from types import SimpleNamespace

from agent_revamp.preprocess.tool_sanitizer import (
    inject_account_id,
    sanitize_tool_schema,
    translate_tool_args,
)


def _tool(name, properties=None, required=None, description="desc", input_schema=...):
    schema = input_schema if input_schema is not ... else {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


def test_hidden_account_id_dropped_from_properties_and_required():
    tool = _tool(
        "execute_query",
        properties={"account_id": {"type": "integer"}, "sql": {"type": "string"}},
        required=["account_id", "sql"],
    )
    out = sanitize_tool_schema(tool)
    params = out["function"]["parameters"]
    assert "account_id" not in params["properties"]
    assert params["properties"].keys() == {"sql"}
    assert params["required"] == ["sql"]


def test_generate_report_hidden_account_id_dropped_other_params_kept():
    tool = _tool(
        "generate_report",
        properties={
            "account_id": {"type": "integer"},
            "sql": {"type": "string"},
            "title": {"type": "string"},
            "viz_type": {"type": "string"},
        },
        required=["account_id", "sql"],
    )
    out = sanitize_tool_schema(tool)
    params = out["function"]["parameters"]
    assert params["properties"].keys() == {"sql", "title", "viz_type"}
    assert params["required"] == ["sql"]


def test_description_with_a_rewrite_entry_ignores_the_original_entirely():
    """sanitize_tool_schema doesn't scrub the incoming MCP description -- for any tool
    name listed in tool_description_rewrites, the original is discarded wholesale and
    replaced by the fixed, pre-written safe text, regardless of what the original said."""
    leaky_original = "real desc with account_id and `transaction` and db://schema and uid"
    tool = _tool("execute_query", description=leaky_original)
    out = sanitize_tool_schema(tool)
    description = out["function"]["description"]
    assert description != leaky_original
    for leak in ("account_id", "`transaction`", "db://schema"):
        assert leak not in description
    assert "friendly" in description.lower()


def test_unmapped_tool_keeps_original_description_and_properties_verbatim():
    tool = _tool(
        "search_market_data",
        properties={"query": {"type": "string"}},
        description="Search the web for market data.",
    )
    out = sanitize_tool_schema(tool)
    assert out["function"]["name"] == "search_market_data"
    assert out["function"]["description"] == "Search the web for market data."
    assert out["function"]["parameters"]["properties"] == {"query": {"type": "string"}}


def test_input_schema_none_is_tolerated():
    tool = _tool("execute_query", input_schema=None, description="d")
    out = sanitize_tool_schema(tool)
    assert out["function"]["parameters"]["properties"] == {}


def test_translate_tool_args_is_identity_passthrough():
    """No tool_arg_maps exist in agent-revamp's schema_map.json, so translate_tool_args
    (friendly -> real param names) is a no-op identity mapping today -- unlike the
    reference repo, which renames several create_*/list_* tool arguments."""
    args = {"sql": "SELECT 1", "title": "Report", "viz_type": "bar"}
    assert translate_tool_args("generate_report", args) == args


def test_inject_account_id_overwrites_whatever_the_model_supplied():
    args = {"sql": "SELECT 1", "account_id": 999}
    out = inject_account_id("execute_query", args, account_id=42)
    assert out["account_id"] == 42


def test_inject_account_id_defaults_when_no_caller_identity():
    """Mirrors the CLI path (no per-user auth) falling back to settings.default_account_id."""
    out = inject_account_id("execute_query", {"sql": "SELECT 1"}, account_id=None)
    from agent_revamp.config import settings

    assert out["account_id"] == settings.default_account_id


def test_inject_account_id_is_noop_for_a_tool_that_never_hides_it():
    args = {"query": "market rates"}
    assert inject_account_id("search_market_data", args, account_id=42) == args
