"""Written from scratch. realbooks-agents has a conceptually similar but structurally
different mechanism (test_permission_guard.py's role/permission-based filter_tools);
agent-revamp's process_class.py is a fixed per-process-class allowlist instead, so this
doesn't port -- it's tested directly against the real PROCESS_CLASS_REGISTRY.

This is the diagram's "Validate tools and skills for the agent" box and a hard security
boundary agent.py depends on (filter_tools applied once at MCP connect time,
validate_tool_scope re-checked every turn) -- worth pinning down precisely.
"""

from types import SimpleNamespace

import pytest

from agent_revamp.preprocess.process_class import (
    PROCESS_CLASS_REGISTRY,
    ToolScopeError,
    allowed_tools,
    filter_tools,
    validate_tool_scope,
)


def _tool(name):
    return SimpleNamespace(name=name)


def test_allowed_tools_returns_the_real_registry_set_for_penny():
    assert allowed_tools("penny") == {"execute_query", "generate_report"}


def test_allowed_tools_raises_for_an_unregistered_process_class():
    with pytest.raises(ToolScopeError, match="Unknown process class"):
        allowed_tools("nonexistent_class")


def test_filter_tools_keeps_only_allowlisted_names_for_penny():
    tools = [_tool("execute_query"), _tool("generate_report"), _tool("create_asset"), _tool("stage_rows")]
    filtered = filter_tools(tools, "penny")
    assert {t.name for t in filtered} == {"execute_query", "generate_report"}


def test_filter_tools_returns_empty_list_when_nothing_matches():
    tools = [_tool("stage_rows"), _tool("promote_rows")]
    assert filter_tools(tools, "penny") == []


def test_validate_tool_scope_passes_silently_for_in_scope_names():
    validate_tool_scope(["execute_query", "generate_report"], "penny")  # must not raise


def test_validate_tool_scope_raises_on_any_out_of_scope_name():
    """The security-critical case: even ONE out-of-scope name in an otherwise valid list
    must refuse the whole turn -- this is the check agent.py relies on as defense in
    depth against a stale/cross-process-class Qdrant hit ever reaching the model."""
    with pytest.raises(ToolScopeError, match="stage_rows"):
        validate_tool_scope(["execute_query", "stage_rows"], "penny")


def test_validate_tool_scope_with_empty_list_never_raises():
    validate_tool_scope([], "penny")  # the "nothing to send this turn" case is always safe


@pytest.mark.parametrize("process_class", list(PROCESS_CLASS_REGISTRY.keys()))
def test_every_registered_process_class_has_a_non_empty_allowlist(process_class):
    """Registry-shape guard: a process class with an empty allowed_tools set would
    silently mean "no tools ever," which is always a config mistake, not a real state."""
    assert allowed_tools(process_class), f"{process_class} has an empty allowlist"


def test_process_classes_do_not_share_tool_names_across_registries_unexpectedly():
    """execute_query/generate_report (penny) must not appear in a process class that was
    never meant to have DB read access at all (transaction_saving, the statement-import
    worker) -- guards against a copy-paste registry mistake."""
    assert "execute_query" not in allowed_tools("transaction_saving")
    assert "generate_report" not in allowed_tools("transaction_saving")
