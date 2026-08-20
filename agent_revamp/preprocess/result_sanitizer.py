"""Result sanitizer: strips real column names from MCP tool results.

Ported from realbooks-agents/app/agent/result_sanitizer.py, trimmed to penny-mcp's two result
shapes ("list" for execute_query, "report" for generate_report — no "create"/"profile" types
exist here). After a tool executes, the raw result contains real DB column names as JSON keys
(e.g. `{"amount": 500, "flow_type": "expense"}`). This module replaces real keys with friendly
names and drops hidden/internal columns, so the LLM never sees raw column names in data.

Deviation from the reference implementation: generate_report's underlying service
(app/services/metabase.py::create_embed_report) returns a top-level "sql" key containing the
real, translated SQL (with real table/column names) alongside the safe metadata fields. The
reference's "report" branch passes through any key not in keep_keys unchanged, which would
leak that real SQL straight back into the model's context — the exact thing this whole layer
exists to prevent. Here, only keys explicitly listed in keep_keys (plus the recursed data_key)
survive; everything else, including "sql", is dropped rather than passed through.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SCHEMA_MAP_PATH = Path(__file__).parent / "schema_map.json"

# Unwraps function-wrapped result keys like "SUM(amount)" so the inner real column name can
# be renamed ("SUM(Amount)").
_FUNC_WRAP_RE = re.compile(r"^[A-Z_]+\((.*)\)$")

_BUILTIN_PASS_THROUGH_KEYS: set[str] = {"error", "info"}

# Non-printable/control characters (keeping \n and \t) and a length cap applied to every
# free-text value (Vendor, Description, ...) returned from the database before it re-enters
# the model's context — a defense-in-depth bound against instructions smuggled into
# user-entered data (prompt injection via tool output). This is independent of, and doesn't
# rely on, the model correctly treating tool output as data rather than instructions.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_STRING_LEN = 500


def _sanitize_string(value: str) -> str:
    cleaned = _CONTROL_CHAR_RE.sub("", value)
    if len(cleaned) > _MAX_STRING_LEN:
        cleaned = cleaned[:_MAX_STRING_LEN] + "...[truncated]"
    return cleaned


def _load_map() -> dict[str, Any]:
    with open(_SCHEMA_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _build_reverse_column_map(tool_config: dict | None) -> dict[str, str] | None:
    if not tool_config or not tool_config.get("column_map"):
        return None
    return tool_config["column_map"]  # real → friendly, already defined this way


def _build_hidden_set() -> set[str]:
    schema_map = _load_map()
    hidden: set[str] = set()
    for tdef in schema_map.get("tables", {}).values():
        for h in tdef.get("hidden_columns", []):
            hidden.add(h.lower())
    return hidden


def _build_backstop_map() -> dict[str, str]:
    """Global real → friendly map from the schema tables (lower-cased keys).

    A real column that maps to different friendly names across tables is dropped: it can't
    be disambiguated from a result key alone, and the SQL translator already re-aliases those
    columns to the right friendly name in the SELECT list.
    """
    schema_map = _load_map()
    mapping: dict[str, str] = {}

    def _add(real: str, friendly: str) -> None:
        key = real.lower()
        existing = mapping.get(key)
        if existing is None:
            mapping[key] = friendly
        elif existing != friendly:
            mapping.pop(key, None)

    for tdef in schema_map.get("tables", {}).values():
        for fcol, rcol in tdef.get("columns", {}).items():
            _add(rcol, fcol)
        _add(tdef.get("pk_real", "uid"), tdef.get("pk_column", "Id"))
    return mapping


def _rename_key(key: str, backstop: dict[str, str]) -> str:
    lower = key.lower()
    if lower in backstop:
        return backstop[lower]
    wrapped = _FUNC_WRAP_RE.match(key)
    if wrapped and wrapped.group(1).lower() in backstop:
        return backstop[wrapped.group(1).lower()]
    return key


def _sanitize_object(
    obj: dict[str, Any],
    column_map: dict[str, str] | None,
    hidden: set[str],
    keep_keys: set[str],
    backstop: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in obj.items():
        lower_key = key.lower()
        if lower_key in hidden:
            continue
        if isinstance(value, str):
            value = _sanitize_string(value)
        if lower_key in keep_keys or key in keep_keys:
            result[key] = value
            continue
        friendly = column_map.get(key) if column_map else None
        if friendly:
            result[friendly] = value
        elif column_map is None:
            result[_rename_key(key, backstop or {})] = value
        else:
            continue
    return result


def sanitize_tool_result(content: str, tool_name: str) -> str:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content

    schema_map = _load_map()
    tool_config = schema_map.get("result_tools", {}).get(tool_name)
    column_map = _build_reverse_column_map(tool_config)
    hidden = _build_hidden_set()
    backstop = _build_backstop_map()

    inner_keys = _BUILTIN_PASS_THROUGH_KEYS.copy()
    tool_type = tool_config.get("type") if tool_config else None

    def _recurse(value: Any) -> Any:
        if isinstance(value, dict):
            return _sanitize_object(value, column_map, hidden, inner_keys, backstop)
        if isinstance(value, list):
            return [_recurse(item) for item in value]
        return value

    if tool_type == "list" and isinstance(data, list):
        data = [_recurse(row) for row in data]
    elif tool_type == "list" and isinstance(data, dict):
        root_key = tool_config.get("root_key", "rows")
        rows = data.get(root_key, [])
        if isinstance(rows, list):
            data[root_key] = [_recurse(row) for row in rows]
    elif tool_type == "report" and isinstance(data, dict):
        data_key = tool_config.get("data_key", "data_preview")
        keep = set(tool_config.get("keep_keys", []))
        result = {}
        for k, v in data.items():
            if k in keep or k.lower() in keep:
                result[k] = v
            elif k == data_key and isinstance(v, list):
                result[k] = [_recurse(row) for row in v]
            # Any other key (e.g. the real "sql" the service returns alongside the safe
            # metadata) is intentionally dropped, not passed through — see module docstring.
        data = result
    elif isinstance(data, dict):
        data = _recurse(data)

    return json.dumps(data, ensure_ascii=False)
