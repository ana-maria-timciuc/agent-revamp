"""Ported from realbooks-agents/tests/test_schema_leak_guard.py's TestSanitizeUserText
class -- the single most direct port in this suite, since leak_guard.py's own docstring
says it's "the regex pipeline below is unchanged in approach" from
harness.py::_sanitize_user_text. Adapted to agent-revamp's own table/column vocabulary
(schema_map.json). agent-revamp's _SCHEMA_WORDS_RE is built dynamically from
schema_map.json (real_schema_words()) rather than hand-maintained like the original
repo's, so generic SQL/product-name mentions like "MariaDB" needed a small, separate,
explicitly hand-maintained supplement (_INTERNAL_PRODUCT_NAMES_RE) to be caught outside a
full SELECT...FROM statement too -- see
test_bare_internal_product_names_are_caught_outside_a_statement_too below. Every other
case ported directly since the underlying regex *behaviors* (dot-notation, SQL-statement
detection, tax-ID formats, meta phrases, PK-line dumps, and the negative/false-positive
guards) are schema-agnostic.

Ground-truth outputs were captured by running sanitize_user_text() directly against this
repo's actual schema_map.json.
"""

import json

from agent_revamp.postprocess.leak_guard import redact_real_schema_leaks, sanitize_user_text


def test_redacts_dot_notation_with_friendly_keys():
    text = (
        "Data flows via Properties.Id -> Transactions.PropertyId, "
        "Loans.PropertyId -> Properties.Id, and Projects.ParentProjectId -> Projects.Id."
    )
    out = sanitize_user_text(text)
    for leak in (
        "Properties.Id",
        "Transactions.PropertyId",
        "Loans.PropertyId",
        "Projects.ParentProjectId",
        "Projects.Id",
    ):
        assert leak not in out


def test_redacts_full_sql_statement_including_embedded_function_names():
    """CONVERT_TZ/MariaDB are only caught when they appear inside a genuine
    SELECT...FROM statement (via the SQL-keyword/function-paren detector) -- see the
    module docstring above for why a bare, statement-free mention differs."""
    text = (
        "Try SELECT CONVERT_TZ(Date, '+00:00', '+03:00') AS local_time "
        "FROM Transactions to get local time on MariaDB."
    )
    out = sanitize_user_text(text)
    assert "CONVERT_TZ" not in out
    assert "MariaDB" not in out
    assert "FROM Transactions" not in out


def test_bare_internal_product_names_are_caught_outside_a_statement_too():
    """_SCHEMA_WORDS_RE is schema-derived and can't know about "MariaDB" (it's not a
    column/table identifier), so a small hand-maintained supplement
    (_INTERNAL_PRODUCT_NAMES_RE) catches it unconditionally, not just inside a full SQL
    statement."""
    out = sanitize_user_text("The database is MariaDB.")
    assert "MariaDB" not in out
    assert "database" not in out.lower()


def test_redacts_plural_meta_phrases():
    text = (
        "Primary keys (all are named Id):\n"
        "Foreign keys and links:\n"
        "join paths:\n"
        "The key columns are important."
    )
    out = sanitize_user_text(text)
    for leak in ("Primary keys", "Foreign keys", "join paths", "key columns"):
        assert leak not in out


def test_redacts_category_dot_notation():
    text = "Transactions.Category -> CategoryLookup.Id, CategoryLookup.CategoryName = 'Utilities'"
    out = sanitize_user_text(text)
    for leak in ("Transactions.Category", "CategoryLookup.Id", "CategoryLookup.CategoryName"):
        assert leak not in out


def test_redacts_pk_table_lines_but_leaves_ordinary_label_value_lines_alone():
    text = "    Transactions: Id\n    Properties: Id\nIncome: $1,234.56"
    out = sanitize_user_text(text)
    assert "Transactions: Id" not in out
    assert "Properties: Id" not in out
    assert "Income: $1,234.56" in out  # ordinary "Label: value" prose must survive


def test_redacts_ein_ssn_itin_formats():
    assert "12-3456789" not in sanitize_user_text("The contractor's EIN is 12-3456789.")
    assert "123-45-6789" not in sanitize_user_text("Their SSN is 123-45-6789.")
    assert "912-70-1234" not in sanitize_user_text("ITIN 912-70-1234 was provided.")


def test_tax_id_redaction_leaves_phone_numbers_dates_and_money_alone():
    """Negative test: proves the EIN/SSN regex doesn't false-positive on phone numbers,
    ISO/US dates, or currency amounts -- these must come back byte-identical."""
    for safe in (
        "Call the office at 303-555-0148.",
        "Signed on 2026-08-10 and again on 09-14-2026.",
        "The bill came to $123,456.78.",
    ):
        assert sanitize_user_text(safe) == safe


def test_friendly_catalog_labels_are_expanded_not_blanked():
    """PropertyName/PurchasePrice are internal catalog labels (PascalCase), but the
    underlying meaning ("what fields do I need to create a property?") is a legitimate
    answer -- they get humanized to plain words rather than redacted outright."""
    out = sanitize_user_text(
        "What fields do I need to create a property? PropertyName, PurchasePrice, and ARV."
    )
    assert "PropertyName" not in out
    assert "Property Name" in out
    assert "Purchase Price" in out


def test_ordinary_conversation_with_generic_column_words_is_untouched():
    """"select"/"vendor" as plain English verbs/nouns must never get mangled just
    because they also happen to be SQL keywords / a real column name elsewhere."""
    safe = "select the report from the list, then select a vendor from the dropdown"
    assert sanitize_user_text(safe) == safe


def test_empty_string_returns_unchanged():
    assert sanitize_user_text("") == ""


def test_redact_real_schema_leaks_substitutes_friendly_name_when_one_exists():
    """The universal method (also used on raw tool-result content, see
    test_dispatch-level coverage in test_agent_chat.py) replaces a caught real name with
    its friendly equivalent rather than opaquely redacting it, so the text stays
    informative."""
    assert redact_real_schema_leaks("the column is called flow_type") == "the column is called Type"


def test_redact_real_schema_leaks_falls_back_to_redaction_with_no_friendly_form():
    """account_id/delete_timestamp/etc. are hidden columns with no friendly equivalent at
    all -- there's nothing safe to substitute, so they still fall back to full redaction."""
    out = redact_real_schema_leaks("filter by account_id")
    assert "account_id" not in out
    assert "[details omitted]" in out


def test_redact_real_schema_leaks_never_touches_friendly_pascalcase_keys():
    """Must be safe to run on JSON tool-result content: unlike sanitize_user_text(), it
    must never "humanize" a friendly catalog label, since that would corrupt a JSON key's
    exact spelling (e.g. "PropertyName" -> "Property Name" breaks the key the model needs
    to keep referencing consistently)."""
    payload = json.dumps({"PropertyName": "Oak St", "Amount": 500})
    out = redact_real_schema_leaks(payload)
    assert out == payload
    assert json.loads(out) == {"PropertyName": "Oak St", "Amount": 500}


def test_redact_real_schema_leaks_closes_the_execute_query_error_leak():
    """Regression test for the exact leak found in review: a raw DB/driver error from a
    "list"-type tool (execute_query) used to pass through sanitize_tool_result's
    allow-list untouched. redact_real_schema_leaks() is now applied in
    Agent._dispatch() right after sanitize_tool_result(), closing that gap."""
    leaked = json.dumps(
        {"error": "Unknown column 'flow_type' in 'field list' on table `transaction`.account_id=45", "rows": []}
    )
    out = redact_real_schema_leaks(leaked)
    assert "flow_type" not in out
    assert "account_id" not in out
    assert "Type" in out  # substituted, not just blanked
    assert json.loads(out)["rows"] == []  # JSON structure survives
