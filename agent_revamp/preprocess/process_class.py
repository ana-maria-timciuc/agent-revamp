"""ProcessClass: which call-type an Agent instance is running as, and which MCP tools it's
allowed to see. Mirrors realbooks-agents/app/agent/registry.py::AGENT_REGISTRY — the same
four call types across the RealBooks agent ecosystem:

- "penny", "dollar_bill", "uncle_sam" — the existing chat agents (registry entries here are
  verified against the real dollar-bill-mcp/uncle-sam-mcp server code, not just copied from
  realbooks-agents' registry).
- "transaction_saving" — the new bank-statement-import worker (see
  E:\\realbooks\\worker_agents_notes.md and the worker-agent architecture diagram). Its MCP
  server isn't built yet; allowed_tools is inferred from the diagram's "Statement-Import MCP"
  arrow label ("stage_rows + promote_rows") and is expected to change once that server exists.

Only "penny" is actually wired up in agent-revamp today — the only MCP connection this
project has is penny-mcp. The other three are registered here as data so the tool-scope
validation mechanism is complete and ready the moment those MCP connections are added;
activating one is then just pointing mcp_url at the right server, not touching this file.

Two independent enforcement points, matching realbooks-agents' pattern plus one addition:
- filter_tools(): hard-removes any fetched MCP tool outside the allowlist, applied once
  right after list_tools() — mirrors harness.py's `filtered = [t for t in mcp_tools if
  allowed is None or t.name in allowed]`.
- validate_tool_scope(): a second, independent check applied right before every completion
  call, asserting every tool name about to be shown to the model is still within the
  allowlist. filter_tools() should already guarantee this; this exists specifically to catch
  agent_revamp's Qdrant-based tool retrieval (_retrieve_tools) surfacing something out of
  scope — e.g. a shared QDRANT_TOOLS_COLLECTION carrying stale entries from a session run
  under a different process class (ToolIndex.index_tools() upserts but never prunes, unlike
  SkillIndex). Agent._retrieve_tools already guards against that via a self._tool_schemas
  membership check, so this is defense in depth against that guard ever being removed or
  bypassed, not a currently-exploitable gap — but it's exactly the kind of invariant worth
  asserting explicitly rather than relying on incidentally being true elsewhere.
"""

from __future__ import annotations

from typing import Literal

ProcessClass = Literal["transaction_saving", "penny", "uncle_sam", "dollar_bill"]

PROCESS_CLASS_REGISTRY: dict[str, dict] = {
    "penny": {
        "allowed_tools": {"execute_query", "generate_report"},
    },
    "dollar_bill": {
        # Verified 2026-08-20 against E:\realbooks\realbooks-dollar-bill-mcp's actual tool
        # registrations (app/tools/*.py, app/resources/schema.py) — exact match, no stale entries.
        "allowed_tools": {
            "list_assets",
            "list_entities",
            "list_transactions",
            "list_projects",
            "list_rooms",
            "list_loans",
            "list_contractors",
            "execute_query",
            "get_database_schema",
            "get_account_profile",
            "create_asset",
            "create_entity",
            "create_project",
            "create_sub_project",
            "create_loan",
            "create_loan_payment",
            "create_transaction",
            "create_contractor",
        },
    },
    "uncle_sam": {
        # Verified 2026-08-20 against E:\realbooks\realbooks-uncle-sam-mcp's actual tool
        # registrations (app/tools/search_tools.py) — exact match.
        "allowed_tools": {"search_market_data", "fetch_page_content"},
    },
    "transaction_saving": {
        # TODO(transaction-saving-mcp): PLACEHOLDER — Statement-Import MCP doesn't exist yet.
        # allowed_tools is inferred from the worker-agent architecture diagram's MCP arrow
        # label, not read from real server code (unlike penny/dollar_bill/uncle_sam above).
        # Once that MCP server is built: (1) confirm its real tool names here, (2) add a
        # statement_import_mcp_url setting in config.py alongside penny_mcp_url, and (3) wire
        # a connection for this process class in Agent (today Agent only ever connects to
        # penny_mcp_url regardless of process_class).
        "allowed_tools": {"stage_rows", "promote_rows"},
    },
}


class ToolScopeError(Exception):
    """Raised when a tool outside the current process class's allowlist would reach (or did
    reach) the model. Callers must treat this as a refused turn, never a warning to log past."""


def allowed_tools(process_class: ProcessClass) -> set[str]:
    entry = PROCESS_CLASS_REGISTRY.get(process_class)
    if entry is None:
        raise ToolScopeError(f"Unknown process class: {process_class!r}")
    return entry["allowed_tools"]


def filter_tools(mcp_tools: list, process_class: ProcessClass) -> list:
    """Hard-remove any fetched MCP tool the process class isn't allowed to use at all."""
    allowed = allowed_tools(process_class)
    return [t for t in mcp_tools if t.name in allowed]


def validate_tool_scope(tool_names: list[str], process_class: ProcessClass) -> None:
    """Assert every tool name about to be shown to the model this turn is within the
    process class's allowlist. Raises ToolScopeError (never silently drops) — a tool
    appearing here at all means filtering was bypassed or retrieval is misconfigured,
    which is worth failing loudly on rather than quietly filtering and moving on."""
    allowed = allowed_tools(process_class)
    out_of_scope = set(tool_names) - allowed
    if out_of_scope:
        raise ToolScopeError(
            f"Tool(s) {sorted(out_of_scope)} are outside the {process_class!r} process "
            "class's allowed set — refusing to send this turn to the model."
        )
