"""Ported from realbooks-agents/tests/test_sql_translator.py, adapted to agent-revamp's
own (smaller) schema_map.json: tables Transactions->`transaction` (backtick-quoted),
Properties->asset, Entities->entities, Loans->loan, Projects->project,
CategoryLookup->statuses_types (no mandatory_where at all — it's a lookup table, not
tenant-scoped). agent-revamp's translate_sql(sql, account_id) has no tz_offset
parameter (unlike the reference it was ported from), so timezone-conversion cases were
dropped rather than adapted.

Ground-truth outputs below were captured by running translate_sql() directly against
this repo's actual schema_map.json, not assumed from the original test file.
"""

import pytest

from agent_revamp.preprocess.sql_translator import SQLTranslationError, translate_sql


def test_translates_friendly_names_and_adds_mandatory_filters():
    out = translate_sql("SELECT Amount FROM Transactions WHERE Type = 'income'", account_id=7)
    assert "amount AS Amount" in out
    assert "`transaction`" in out  # Transactions.backtick = true
    assert "flow_type = 'income'" in out
    assert "account_id = 7" in out
    assert "delete_timestamp IS NULL" in out
    assert "is_archived" in out
    assert "split_transactions_id IS NULL" in out


def test_aliases_bare_selected_columns_to_friendly_names():
    out = translate_sql("SELECT Amount, Id FROM Transactions", account_id=7)
    assert "amount AS Amount" in out
    assert "uid AS Id" in out


def test_aliases_qualified_bare_columns_and_join_columns():
    out = translate_sql(
        "SELECT PropertyName, e.EntityName FROM Properties p LEFT JOIN Entities e ON e.Id = p.EntityId",
        account_id=7,
    )
    assert "display_name AS PropertyName" in out
    assert "e.name AS EntityName" in out
    assert "LEFT JOIN entities AS e" in out
    assert "e.uid = p.entity_id" in out  # join condition columns translated too


def test_llm_aliases_are_preserved_and_not_double_aliased():
    out = translate_sql(
        "SELECT t.Amount AS total, SUM(t.Amount) AS income FROM Transactions t", account_id=7
    )
    assert "t.amount AS total" in out
    assert "SUM(t.amount) AS income" in out
    assert "AS total AS total" not in out
    assert "AS income AS income" not in out


def test_aggregates_without_alias_are_left_untouched():
    out = translate_sql("SELECT SUM(Amount), COUNT(*) FROM Transactions", account_id=7)
    assert "SUM(amount)" in out
    assert "COUNT(*)" in out


def test_star_select_not_aliased():
    out = translate_sql("SELECT * FROM Properties", account_id=7)
    assert "SELECT *" in out


def test_group_by_and_order_by_by_column_not_aliased():
    out = translate_sql(
        "SELECT Category, SUM(Amount) AS total FROM Transactions GROUP BY Category ORDER BY Amount DESC",
        account_id=7,
    )
    assert "GROUP BY sub_category_id" in out
    assert "ORDER BY amount DESC" in out


def test_category_lookup_table_has_no_mandatory_filters():
    """CategoryLookup's mandatory_where is [] in schema_map.json (it's a shared lookup
    table, not tenant-scoped) — confirms translate_sql doesn't force account scoping
    onto tables that were never configured to need it."""
    out = translate_sql("SELECT CategoryName FROM CategoryLookup", account_id=7)
    assert out == "SELECT name AS CategoryName FROM statuses_types"
    assert "account_id" not in out


def test_non_friendly_sql_passes_through_translation():
    assert translate_sql("SELECT 1", account_id=7) == "SELECT 1"


def test_unresolved_table_and_column_raise():
    with pytest.raises(SQLTranslationError, match="Unknown table/column"):
        translate_sql("SELECT Foo FROM Bar", account_id=7)


def test_non_select_statement_rejected():
    with pytest.raises(SQLTranslationError, match="Only SELECT queries are allowed"):
        translate_sql("DELETE FROM Transactions", account_id=7)


def test_empty_sql_returns_unchanged():
    assert translate_sql("", account_id=7) == ""
    assert translate_sql("   ", account_id=7) == "   "


def test_order_by_alias_reference_resolves():
    """Was a bug (raised SQLTranslationError): a bare ORDER BY reference to a
    SELECT-list alias (not a real friendly column) is now left untouched instead of
    being treated as unresolved."""
    out = translate_sql(
        "SELECT Category, SUM(Amount) AS total FROM Transactions GROUP BY Category ORDER BY total DESC",
        account_id=7,
    )
    assert "GROUP BY sub_category_id" in out
    assert "ORDER BY total DESC" in out  # "total" is the alias itself -- never translated


def test_subquery_alias_qualified_column_resolves():
    """Was a bug (raised SQLTranslationError): a column qualified by a derived-table
    (subquery) alias now resolves, because the subquery's own inner translation pass
    already re-aliases its output to the same friendly name the outer query expects."""
    out = translate_sql("SELECT p.PropertyName FROM (SELECT PropertyName FROM Properties) p", account_id=7)
    assert "p.PropertyName" in out  # left untouched -- matches the subquery's own aliased output
    assert "display_name AS PropertyName FROM asset" in out  # inner subquery translated correctly


def test_mandatory_where_for_a_subquery_table_attaches_inside_the_subquery():
    """Was a separate bug (found while fixing the one above, then fixed too):
    _add_mandatory_where used to always attach mandatory filters to the top-level query,
    even for a table nested inside a subquery -- producing invalid SQL that referenced
    `asset` in a scope where it didn't exist. Now each table's mandatory filters attach
    to its own enclosing SELECT."""
    out = translate_sql("SELECT p.PropertyName FROM (SELECT PropertyName FROM Properties) p", account_id=7)
    assert out == (
        "SELECT p.PropertyName FROM (SELECT display_name AS PropertyName FROM asset "
        "WHERE (asset.account_id = 7 AND asset.delete_timestamp IS NULL) AND COALESCE(asset.is_archived, 0) = 0) AS p"
    )
    # No WHERE clause on the outer query -- "asset" is never in scope there.
    assert out.count("WHERE") == 1


def test_mandatory_where_is_scoped_correctly_with_mixed_top_level_and_subquery_tables():
    """A real top-level table and a subquery-derived table in the same statement must
    each get their mandatory filters attached to their OWN scope, not merged into one
    or attached to the wrong one."""
    out = translate_sql(
        "SELECT t.Amount, p.PropertyName FROM Transactions t, (SELECT PropertyName FROM Properties) p",
        account_id=7,
    )
    assert out == (
        "SELECT t.amount AS Amount, p.PropertyName FROM `transaction` AS t, "
        "(SELECT display_name AS PropertyName FROM asset "
        "WHERE (asset.account_id = 7 AND asset.delete_timestamp IS NULL) AND COALESCE(asset.is_archived, 0) = 0) AS p "
        "WHERE ((t.account_id = 7 AND t.delete_timestamp IS NULL) AND COALESCE(t.is_archived, 0) = 0) "
        "AND t.split_transactions_id IS NULL"
    )
