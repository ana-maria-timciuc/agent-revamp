"""Ported from realbooks-agents/tests/test_result_sanitizer.py, trimmed to agent-revamp's
actual two result_tools entries (execute_query: "list"/root_key "rows"; generate_report:
"report"/data_key "data_preview") -- there is no create_asset/list_assets/list_contractors/
create_contractor/get_database_schema config here (penny only has execute_query and
generate_report, per process_class.py's allowed_tools), so those cases were dropped
rather than adapted.

One case genuinely differs from the original repo, not just by omission: `display_name`
is unambiguous in agent-revamp's (smaller) schema -- only Properties.PropertyName maps to
it -- so it IS renamed via the global backstop map here, unlike the original repo where
it was ambiguous across multiple tables and therefore left untouched. Ground-truth
outputs below were captured by running sanitize_tool_result() directly against this
repo's actual schema_map.json.
"""

import json

from agent_revamp.postprocess.result_sanitizer import sanitize_tool_result


def test_execute_query_rows_get_friendly_names_and_drop_hidden_fields():
    content = json.dumps(
        {
            "rows": [
                {
                    "amount": 500,
                    "flow_type": "expense",
                    "account_id": 45,
                    "uid": 3,
                    "delete_timestamp": None,
                    "month": "2024-01",
                }
            ]
        }
    )
    out = json.loads(sanitize_tool_result(content, "execute_query"))
    row = out["rows"][0]
    assert row == {"Amount": 500, "Type": "expense", "month": "2024-01"}


def test_empty_rows_and_error_shapes_survive_unchanged():
    content = json.dumps({"rows": [], "info": "Query returned 0 rows."})
    assert sanitize_tool_result(content, "execute_query") == content


def test_function_wrapped_keys_are_renamed():
    content = json.dumps({"rows": [{"SUM(amount)": 1000, "COUNT(*)": 4}]})
    out = json.loads(sanitize_tool_result(content, "execute_query"))
    assert out["rows"][0] == {"Amount": 1000, "COUNT(*)": 4}


def test_generate_report_data_preview_renamed_top_level_keys_kept():
    content = json.dumps(
        {
            "token": "tok123",
            "question_id": 5,
            "status": "ok",
            "sql": "SELECT * FROM `transaction`",  # must never survive -- see next assertion
            "data_preview": [{"amount": 200, "flow_type": "income"}],
        }
    )
    out = json.loads(sanitize_tool_result(content, "generate_report"))
    assert out["token"] == "tok123"
    assert out["question_id"] == 5
    assert out["status"] == "ok"
    assert "sql" not in out  # real SQL is dropped, not passed through -- see module docstring
    assert out["data_preview"] == [{"Amount": 200, "Type": "income"}]


def test_unambiguous_display_name_is_renamed_via_backstop_map():
    """Unlike the realbooks-agents original (where display_name was ambiguous across
    several tables and therefore left untouched), agent-revamp's schema only has one
    table mapping to display_name (Properties.PropertyName), so the generic backstop
    path in result_sanitizer.py can safely rename it here."""
    content = json.dumps({"rows": [{"display_name": "Oak St"}]})
    out = json.loads(sanitize_tool_result(content, "execute_query"))
    assert out["rows"][0] == {"PropertyName": "Oak St"}


def test_non_json_content_passes_through_unchanged():
    assert sanitize_tool_result("not json", "execute_query") == "not json"


def test_unknown_tool_name_still_applies_the_generic_backstop_rename():
    content = json.dumps({"amount": 10, "account_id": 1})
    out = json.loads(sanitize_tool_result(content, "some_other_tool"))
    assert out == {"Amount": 10}
