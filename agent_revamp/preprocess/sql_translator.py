"""SQL translator: friendly-name SQL → real SQL via sqlglot AST manipulation.

Ported near-verbatim from realbooks-agents/app/agent/sql_translator.py. The LLM writes SQL
using friendly table/column names from schema_map.json. This module parses the SQL, replaces
every identifier with its real counterpart, injects mandatory WHERE clauses (tenant scoping,
soft-delete, archive, split-transaction), and produces valid MySQL ready for penny-mcp.

Deviation from the reference: sqlglot does not automatically quote `transaction` (it isn't in
its MySQL reserved-word list), so the "backtick" flag in schema_map.json is honored explicitly
here via `quoted=True` on the translated table identifier — without it, penny-mcp's own SQL
guard requires the exact backtick-quoted form and unquoted output would be rejected/incorrect.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

_SCHEMA_MAP_PATH = Path(__file__).parent.parent / "schema_map.json"


class SQLTranslationError(Exception):
    """Raised when the model's SQL can't be safely translated to the real schema —
    unparseable, not a SELECT, or referencing a table/column outside the data catalog.
    Callers must treat this as a rejected query, never fall back to the untranslated SQL."""


def _load_map() -> dict[str, Any]:
    with open(_SCHEMA_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _build_lookups(schema_map: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, Any] = schema_map["tables"]

    friendly_table_to_real: dict[str, str] = {}
    friendly_table_config: dict[str, dict] = {}
    friendly_column_to_table: dict[str, str] = {}
    table_column_maps: dict[str, dict[str, str]] = {}
    table_pk: dict[str, str] = {}

    for friendly_name, tdef in tables.items():
        lower_friendly = friendly_name.lower()
        friendly_table_to_real[lower_friendly] = tdef["real_name"]
        friendly_table_config[friendly_name] = tdef
        friendly_table_config[lower_friendly] = tdef

        col_map = {}
        for fcol, rcol in tdef["columns"].items():
            col_map[fcol.lower()] = rcol
            friendly_column_to_table[fcol.lower()] = friendly_name
        pk_friendly = tdef.get("pk_column", "Id")
        pk_real = tdef.get("pk_real", "uid")
        col_map[pk_friendly.lower()] = pk_real
        friendly_column_to_table[pk_friendly.lower()] = friendly_name
        table_column_maps[friendly_name] = col_map
        table_pk[friendly_name] = pk_real

    return {
        "friendly_table_to_real": friendly_table_to_real,
        "friendly_table_config": friendly_table_config,
        "friendly_column_to_table": friendly_column_to_table,
        "table_column_maps": table_column_maps,
        "table_pk": table_pk,
    }


def _token_is_identifier(node: exp.Expression) -> bool:
    return isinstance(node, exp.Identifier | exp.Var)


def _translate_table(
    node: exp.Table,
    lookups: dict[str, Any],
    referenced: set[str],
    alias_map: dict[str, str],
    unresolved: set[str],
) -> None:
    if not _token_is_identifier(node.this):
        return

    friendly = node.this.this
    if not isinstance(friendly, str):
        return
    fkey = friendly.lower()

    config = lookups["friendly_table_config"].get(fkey) or lookups["friendly_table_config"].get(friendly)
    if config is None:
        unresolved.add(friendly)
        return

    real_name = config["real_name"]
    node.set("this", exp.Identifier(this=real_name, quoted=bool(config.get("backtick"))))

    alias = node.alias_or_name
    alias_map[alias.lower()] = config.get("_orig_friendly", friendly)
    referenced.add(config.get("_orig_friendly", friendly))


def _translate_column(
    node: exp.Column,
    lookups: dict[str, Any],
    alias_map: dict[str, str],
    unresolved: set[str],
) -> None:
    if not _token_is_identifier(node.this):
        return

    col_friendly = node.this.this
    if not isinstance(col_friendly, str):
        return
    col_key = col_friendly.lower()

    table_friendly = None
    table_qualifier_display: str | None = None

    if node.table:
        tbl_name = None
        if isinstance(node.table, str):
            tbl_name = node.table
        elif hasattr(node.table, "this"):
            inner = node.table.this
            if isinstance(inner, str):
                tbl_name = inner
        if tbl_name:
            table_qualifier_display = tbl_name
            tbl_key = tbl_name.lower()
            config = lookups["friendly_table_config"].get(tbl_key) or lookups["friendly_table_config"].get(tbl_name)
            if config:
                if isinstance(node.table, str):
                    node.set("table", exp.Identifier(this=config["real_name"]))
                elif hasattr(node.table, "this"):
                    node.table.set("this", exp.Identifier(this=config["real_name"]))
                table_friendly = tbl_name
            else:
                table_friendly = alias_map.get(tbl_key, tbl_name)

    col_maps = lookups["table_column_maps"]
    real_col = None

    if table_friendly and table_friendly in col_maps:
        real_col = col_maps[table_friendly].get(col_key)
    elif table_friendly:
        for tname, cmap in col_maps.items():
            if tname.lower() == table_friendly.lower():
                real_col = cmap.get(col_key)
                if real_col:
                    break
    else:
        table_friendly = lookups["friendly_column_to_table"].get(col_key)
        if table_friendly and table_friendly in col_maps:
            real_col = col_maps[table_friendly].get(col_key)

    if real_col is not None:
        node.set("this", exp.Identifier(this=real_col))
        # Remember the friendly name so the SELECT pass can re-alias the column: the DB
        # returns real column names as result keys, which must not reach the LLM.
        node.meta["friendly_name"] = col_friendly
    else:
        label = f"{table_qualifier_display}.{col_friendly}" if table_qualifier_display else col_friendly
        unresolved.add(label)


def _alias_selected_columns(tree: exp.Expression) -> None:
    """Re-alias translated SELECT columns back to their friendly names.

    The LLM writes `SELECT Amount FROM Transactions`; after translation the query is
    `SELECT amount FROM transaction` and the MCP result key would be the real name `amount`.
    Alias every bare translated column in the SELECT list to its friendly name
    (`amount AS Amount`) so result keys are friendly without guessing in the sanitizer.
    Expressions the LLM already aliased, aggregates, literals, and `*` are left untouched.
    """
    for select in tree.find_all(exp.Select):
        new_expressions = []
        for expr in select.expressions:
            friendly = expr.meta.get("friendly_name") if isinstance(expr, exp.Column) else None
            if friendly and not expr.alias:
                expr = exp.Alias(this=expr, alias=exp.Identifier(this=friendly))
            new_expressions.append(expr)
        select.set("expressions", new_expressions)


def _add_mandatory_where(
    tree: exp.Expression,
    lookups: dict[str, Any],
    alias_map: dict[str, str],
    account_id: int,
) -> None:
    configs = lookups["friendly_table_config"]

    for alias, friendly_name in alias_map.items():
        config = configs.get(friendly_name)
        if not config:
            continue
        for clause in config.get("mandatory_where", []):
            filled = clause.replace("{alias}", alias).replace("{account_id}", str(account_id))
            try:
                parsed = sqlglot.parse_one(filled, dialect="mysql")
                tree = tree.where(parsed, copy=False)
            except Exception:
                logger.debug("Failed to parse mandatory where clause: %s", filled)


def translate_sql(friendly_sql: str, account_id: int) -> str:
    if not friendly_sql or not friendly_sql.strip():
        return friendly_sql

    schema_map = _load_map()
    lookups = _build_lookups(schema_map)

    friendly_sql = friendly_sql.replace("{account_id}", str(account_id))

    try:
        tree = sqlglot.parse_one(friendly_sql, dialect="mysql")
    except Exception as exc:
        raise SQLTranslationError(f"Could not parse SQL: {exc}") from exc

    if not isinstance(tree, exp.Select | exp.Union):
        raise SQLTranslationError("Only SELECT queries are allowed.")

    alias_map: dict[str, str] = {}
    referenced_tables: set[str] = set()
    unresolved: set[str] = set()

    for table_node in tree.find_all(exp.Table):
        _translate_table(table_node, lookups, referenced_tables, alias_map, unresolved)

    for col_node in tree.find_all(exp.Column):
        _translate_column(col_node, lookups, alias_map, unresolved)

    if unresolved:
        raise SQLTranslationError(
            "Unknown table/column in your query: " + ", ".join(sorted(unresolved))
            + ". Only use names from the data catalog."
        )

    _alias_selected_columns(tree)
    _add_mandatory_where(tree, lookups, alias_map, account_id)

    return tree.sql(dialect="mysql", pretty=False)
