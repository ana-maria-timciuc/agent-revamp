"""Friendly schema builder: replaces raw SQL DDL with a business-language description.

Ported from realbooks-agents/app/agent/schema_mapper.py::build_friendly_schema/build_schema_prompt.
The old flow injected the raw `db://schema` MCP resource (real table/column names) directly
into the system prompt. This module produces a clean description of available data using
ONLY friendly names from schema_map.json — no real column names, no internal mechanics.

The LLM uses this to understand what data exists and to write SQL using friendly table/column
names; sql_translator.py converts those to real SQL before execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_MAP_PATH = Path(__file__).parent / "schema_map.json"


def _load_map() -> dict[str, Any]:
    with open(_SCHEMA_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_COLUMN_HINTS: dict[str, str] = {
    "Amount": "Dollar amount (always positive)",
    "Date": "Transaction date",
    "Type": "One of: 'income', 'expense', 'refund'",
    "Category": "Category ID (join CategoryLookup for name)",
    "CategoryGroup": "One of: 'operational', 'direct'",
    "Description": "Free-text description",
    "Vendor": "Shop or vendor name",
    "PropertyId": "Links to Properties.Id",
    "EntityId": "Links to Entities.Id",
    "ProjectId": "Links to Projects.Id",
    "LoanId": "Links to Loans.Id (set only for loan payments)",
    "PrincipalPaid": "Principal portion of loan payment",
    "InterestPaid": "Interest portion of loan payment",
    "EscrowPaid": "Escrow portion of loan payment",
    "PMIPaid": "PMI portion of loan payment",
    "LateFees": "Late fee portion of loan payment",
    "PropertyName": "Display name of the property",
    "Address": "Street address",
    "City": "City",
    "State": "State",
    "Zip": "ZIP code",
    "PurchasePrice": "Purchase price",
    "PurchaseDate": "Purchase date (YYYY-MM-DD)",
    "ARV": "After-repair value",
    "PropertyType": "Asset type (e.g. long_term_rental, fix_and_flip)",
    "PropertySubType": "Property sub-type (e.g. single_family_home, duplex)",
    "EntityName": "Display name of the entity",
    "LegalType": "Legal type (llc, s_corp, c_corp, llp, sole_proprietor, individual)",
    "PortfolioType": "Portfolio type (long_term_rental, short_term_rental, fix_and_flip, mixed)",
    "Status": "active or closed",
    "Lender": "Lender name",
    "LoanNumber": "Loan number/identifier",
    "LoanPosition": "Loan position (1 = first, 2 = second, etc.)",
    "LoanType": "Loan type (standard, heloc, hard_money, portfolio, interest_only)",
    "LoanTerm": "Loan term in months",
    "Rate": "Interest rate (decimal, e.g. 0.065 = 6.5%)",
    "PrincipalAmount": "Original principal amount",
    "PaymentAmount": "Monthly payment amount",
    "OriginationDate": "Loan origination date",
    "ProjectName": "Project or room display name",
    "Budget": "Total budget",
    "ParentProjectId": "Set for rooms/sub-projects — links to the parent Projects.Id; NULL for main projects",
    "CategoryName": "Human-readable category name",
    "CategoryType": "income_category, operational_expenses, or direct_expenses",
    "EntityModel": "NULL for general categories; 'loan' for loan-payment categories",
}


def _column_hint(friendly_name: str, table: str) -> str:
    return _COLUMN_HINTS.get(friendly_name, f"Column in {table}")


def build_friendly_schema() -> dict[str, Any]:
    """Produce a dict describing available data in business terms — friendly names only."""
    schema_map = _load_map()
    tables = schema_map.get("tables", {})
    joins = schema_map.get("joins", {})
    rules = schema_map.get("sql_rules", {})

    table_list = []
    for friendly_name, tdef in tables.items():
        entry = {
            "table": friendly_name,
            "description": tdef.get("description", ""),
            "id_column": tdef.get("pk_column", "Id"),
            "columns": {},
        }
        for fcol in tdef.get("columns", {}):
            entry["columns"][fcol] = _column_hint(fcol, friendly_name)
        entry["columns"][tdef.get("pk_column", "Id")] = f"Unique record identifier for {friendly_name}"
        table_list.append(entry)

    join_descriptions = [
        {
            "relationship": join_name.replace("→", " → "),
            "from": jdef["from"],
            "to": jdef["to"],
            "type": jdef.get("type", "JOIN"),
        }
        for join_name, jdef in joins.items()
    ]

    return {
        "schema_format": "friendly",
        "data_tables": table_list,
        "join_paths": join_descriptions,
        "sql_rules": rules,
    }


def build_schema_prompt() -> str:
    """Return the text injected into the system prompt (replaces the raw db://schema dump)."""
    data = build_friendly_schema()

    header = (
        "## DATA CATALOG — available in friendly format\n"
        "You have access to the following data. Use ONLY the table and column names "
        "listed here in your SQL queries. They will be automatically translated to "
        "the database before execution. Never guess column names — only use what "
        "is listed below.\n\n"
    )

    tables_text = "### Tables & Columns\n"
    for t in data["data_tables"]:
        tables_text += f"\n**{t['table']}** — {t['description']}\n"
        tables_text += f"  Primary key: `{t['id_column']}`\n"
        for col, hint in t["columns"].items():
            tables_text += f"  - `{col}` — {hint}\n"

    if data["join_paths"]:
        tables_text += "\n### Join Paths\n"
        for j in data["join_paths"]:
            tables_text += f"  - {j['relationship']}: {j['from']} = {j['to']}\n"

    if data["sql_rules"]:
        tables_text += "\n### SQL Rules\n"
        for key, rule in data["sql_rules"].items():
            tables_text += f"  - **{key}**: {rule}\n"

    tables_text += "\n### Mandatory Filters (applied automatically — never write these yourself)\n"
    tables_text += (
        "Account scoping, soft-delete exclusion, archive exclusion, and (for Transactions) "
        "split-transaction exclusion are injected automatically for every table.\n"
    )

    return header + tables_text
