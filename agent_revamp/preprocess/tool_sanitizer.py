"""Friendly tool-schema sanitizer: hides real DB column/arg names from MCP tool definitions.

Ported from realbooks-agents/app/agent/tool_schema_sanitizer.py, driven by `tool_arg_maps`
(unused here — penny-mcp's only args, account_id/sql/title/viz_type, are already business-level
names), `tool_hidden_args`, and `tool_description_rewrites` in schema_map.json.

sanitize_tool_schema() rewrites what the model sees; translate_tool_args() reverses any
argument renaming before the MCP call. inject_account_id() is new here: penny-mcp requires
account_id as a real tool argument, but the model must never control it — it is hidden from
the schema (via tool_hidden_args) and unconditionally injected from settings before dispatch,
the same "tool-execution layer owns identity" pattern realbooks-agents uses (there via a
verified session; here via a single pinned config value since there's no auth layer yet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_revamp.config import settings

_SCHEMA_MAP_PATH = Path(__file__).parent / "schema_map.json"


def _load_map() -> dict[str, Any]:
    with open(_SCHEMA_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _build_arg_maps(
    schema_map: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, set[str]]]:
    arg_maps = schema_map.get("tool_arg_maps", {})
    default = {k.lower(): v for k, v in arg_maps.get("_default", {}).items()}
    per_tool = {
        tool: {k.lower(): v for k, v in tool_map.items()}
        for tool, tool_map in arg_maps.items()
        if tool != "_default"
    }
    hidden = {
        tool: {h.lower() for h in hidden_args}
        for tool, hidden_args in schema_map.get("tool_hidden_args", {}).items()
    }
    return default, per_tool, hidden


def _forward_map(tool_name: str) -> dict[str, str]:
    """real param name → friendly name (lower-cased keys)."""
    default, per_tool, _ = _build_arg_maps(_load_map())
    forward = dict(default)
    forward.update(per_tool.get(tool_name, {}))
    return forward


def _reverse_map(tool_name: str) -> dict[str, str]:
    """friendly name → real param name."""
    return {v: k for k, v in _forward_map(tool_name).items()}


def _hidden_args(tool_name: str) -> set[str]:
    return _build_arg_maps(_load_map())[2].get(tool_name, set())


def sanitize_tool_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP tool definition into an OpenAI tool dict with friendly names.

    Renames parameter keys, drops hidden parameters (e.g. account_id), and replaces the
    description with the friendly rewrite from schema_map.json (falling back to the original).
    """
    schema_map = _load_map()
    forward = _forward_map(tool.name)
    hidden = _hidden_args(tool.name)
    description = schema_map.get("tool_description_rewrites", {}).get(tool.name, tool.description or "")

    params = dict(tool.inputSchema or {"type": "object", "properties": {}})
    properties = params.get("properties", {})
    new_properties: dict[str, Any] = {}
    for key, spec in properties.items():
        if key.lower() in hidden:
            continue
        new_properties[forward.get(key.lower(), key)] = spec
    params["properties"] = new_properties

    if "required" in params:
        params["required"] = [
            forward.get(r.lower(), r) for r in params["required"] if r.lower() not in hidden
        ]

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": params,
        },
    }


def translate_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Translate friendly argument names back to the real MCP parameter names."""
    reverse = _reverse_map(tool_name)
    return {reverse.get(key, key): value for key, value in args.items()}


def inject_account_id(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Overwrite account_id with the pinned config value for tools that hide it.

    The model never sees account_id as a fillable parameter (sanitize_tool_schema drops it
    when listed in tool_hidden_args), so nothing it outputs for that field can be trusted —
    this always replaces it before the call reaches penny-mcp, regardless of what (if
    anything) is present in args.
    """
    if "account_id" in _hidden_args(tool_name):
        args = dict(args)
        args["account_id"] = settings.default_account_id
    return args
