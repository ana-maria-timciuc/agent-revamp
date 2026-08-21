"""Leak guard: strips internal machinery (SQL, schema identifiers, tax IDs) from any text
that might reach the model or the user.

Ported from realbooks-agents/app/agent/harness.py's `_sanitize_user_text` (the regex pipeline
below is unchanged in approach). This is the last line of defense — everything upstream
(tool_sanitizer, sql_translator, result_sanitizer) is meant to keep the model from ever
learning real schema in the first place; this catches whatever slips through anyway (a
hallucinated real name, a recited friendly label, a pasted SQL fragment, a raw DB error
string that made it past result_sanitizer.py's allow-list).

Two entry points, one shared core:
- `redact_real_schema_leaks()` — the universal method: strips/substitutes real schema
  identifiers, raw SQL, tax IDs, and known internal product names. Safe to apply to
  ANY text, including JSON tool-result content, since it never touches friendly
  (PascalCase) catalog labels or generic "schema"/"database" language — those steps are
  prose-specific and would corrupt a JSON key's exact spelling (e.g. turning the key
  `"PropertyName"` into `"Property Name"`) if applied to structured tool output.
  Called from `agent.py::Agent._dispatch()` right after `sanitize_tool_result()`, as
  defense-in-depth against any leak path result_sanitizer.py's per-tool-type allow-lists
  don't cover (its own docstring flags exactly this gap for `"list"`-type tools, e.g. a
  raw DB error surfacing from `execute_query`).
- `sanitize_user_text()` — the assistant-reply-specific superset: everything
  `redact_real_schema_leaks()` does, plus humanizing recited friendly catalog labels and
  stripping generic schema-narration language — safe only for natural-language prose,
  never for structured data.

Not ported: harness.py's `_emit_sanitized`, a streaming-token buffer that holds partial
output across sentence/code-fence boundaries so a multi-token leak isn't missed mid-stream.
agent_revamp's Agent.chat() is non-streaming (the full reply is available at once), so there
is nothing to buffer — sanitize_user_text() below is applied directly to the complete text.
If streaming is added later, port that buffering wrapper too rather than skipping this step.

Deviation from the reference: harness.py's real-schema-word blocklist (_SCHEMA_WORDS_RE) is
hand-maintained and specific to its own (larger) schema. Here it's built dynamically from
schema_map.json via schema_mapper.real_schema_words(), the same "derived from the catalog,
not hand-maintained" approach the reference already uses for schema_dot_notation_words() —
so it stays correct for this project's actual (different, smaller) schema automatically. A
caught real name is substituted with its friendly equivalent (schema_mapper.real_to_friendly_map())
rather than opaquely redacted, wherever one exists — names with no friendly form at all
(hidden columns like account_id, delete_timestamp) still fall back to redaction.

One narrow exception: _INTERNAL_PRODUCT_NAMES_RE below is a small, explicitly
hand-maintained supplement for internal implementation/product names (e.g. "MariaDB")
that will never appear in schema_map.json — real_schema_words() can only ever derive
column/table identifiers from the catalog, not the name of the database engine itself —
so a bare mention needs its own tiny blocklist, redacted regardless of SQL-statement
context (unlike _SCHEMA_WORDS_RE's dot-notation/statement-scoped siblings).
"""

from __future__ import annotations

import re

from agent_revamp.preprocess.schema_mapper import (
    friendly_identifier_words,
    real_schema_words,
    real_to_friendly_map,
    schema_dot_notation_words,
)

_REDACTED = "[details omitted]"

# EIN (NN-NNNNNNN) / SSN & ITIN (NNN-NN-NNNN) — never let a saved tax ID be echoed back,
# even if the model ignores prompt discipline. US phone formats (NNN-NNN-NNNN) and dates
# don't collide with either shape.
_TAX_ID_RE = re.compile(r"\b(?:\d{2}-\d{7}|\d{3}-\d{2}-\d{4})\b")

_SQL_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_SQL_STMT_RE = re.compile(r"\bSELECT\b[\s\S]{0,2000}?\bFROM\b[\s\S]{0,400}", re.I)

# SQL-specific markers that distinguish a real query from innocent "select X from Y" prose.
# When a SELECT ... FROM match also contains one of these, it is unambiguous SQL — redact it
# even when there are no backtick markers. Normal chat ("select the report from the list",
# "select a vendor from the dropdown") passes through untouched. Clause keywords are
# word-bounded; function names require an opening paren and are NOT trailing-bounded, so
# SUM('1') and DATE_TRUNC('month', ...) still match regardless of what follows the paren.
_SQL_KW_RE = re.compile(
    r"\b(?:GROUP\s+BY|ORDER\s+BY|WHERE|HAVING|JOIN|LIMIT|OFFSET|"
    r"UNION\s+(?:ALL\s+)?SELECT|CASE\s+WHEN)\b"
    r"|\b(?:SUM|COUNT|AVG|MIN|MAX|COALESCE|DATE_TRUNC|DATE_FORMAT|CONVERT_TZ)\s*\(",
    re.I,
)

# Distinctive internal identifiers (table.column references). Common English words that
# happen to be column names (date, amount, vendor, ...) are intentionally excluded so normal
# chat text is never mangled. In dot-notation form, however, no friendly name is safe to
# share — "Transactions.Category" or "Properties.Id" never occurs in natural prose — so the
# list is derived from schema_map.json rather than hand-maintained (schema_dot_notation_words).
# Matching is case-sensitive: catalog names match verbatim; lowercase prose or email local
# parts ("first.name@example.com") pass through untouched.
_TABLE_COL_RE = re.compile(
    r"\b\w+\." + "(?:" + "|".join(sorted((re.escape(w) for w in schema_dot_notation_words()), reverse=True)) + r")\b",
)

# Real (non-friendly) table/column identifiers — derived from schema_map.json, see
# schema_mapper.real_schema_words() for what's included/excluded and why.
_SCHEMA_WORDS_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(w) for w in real_schema_words()), reverse=True)) + r")\b",
    re.I,
)

# Real -> friendly substitution table (see schema_mapper.real_to_friendly_map()). A match
# with no entry here (a hidden column with no friendly form at all, e.g. account_id) falls
# back to full redaction — there's nothing safe to reveal instead.
_REAL_TO_FRIENDLY = real_to_friendly_map()


def _replace_real_name(match: re.Match) -> str:
    return _REAL_TO_FRIENDLY.get(match.group(0).lower(), _REDACTED)

# Hand-maintained supplement for internal implementation/product names that can never be
# derived from schema_map.json (it's not a column/table identifier at all) but must never
# reach the user regardless of surrounding context — see module docstring above.
_INTERNAL_PRODUCT_NAMES_RE = re.compile(r"\bMariaDB\b", re.I)

# The friendly catalog (schema_map.json) is for the LLM's own SQL-writing, never for describing
# "how data is stored" to the user — even in safe business-language names. Distinctive PascalCase
# field labels (PropertyName, LoanNumber, ...) never occur in natural prose, so matching them
# verbatim (case-sensitive) catches a recited catalog without touching normal chat.
#
# Instead of dropping those labels outright, they are expanded to their plain-word form
# ("PropertyName" -> "Property Name") so legitimate user-facing content survives — a
# required-fields list ("what do I need to create a property?") legitimately uses exactly
# these catalog labels, and redacting them would blank out the very details the user asked
# for. The camelCase format is what's internal, not the meaning. Bare "Id" (the primary-key
# marker) is NOT expanded — it stays redacted, along with everything structurally revealing.
_FRIENDLY_SCHEMA_WORDS_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(w) for w in friendly_identifier_words()), reverse=True)) + r")\b"
)


def _humanize_friendly_word(word: str) -> str:
    """Expand a PascalCase catalog label to spaced plain words ('PropertyName' -> 'Property Name')."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", word)
    return " ".join(parts)


_FRIENDLY_HUMAN: dict[str, str] = {w: _humanize_friendly_word(w) for w in friendly_identifier_words() if w != "Id"}

# Structural narration of "how the data is organized" — catches a schema description even when
# phrased in the model's own words rather than by reciting a specific table/column name.
# Plurals included ("Primary keys", "Foreign keys", "join paths") — the trailing "s" must not
# defeat the word boundary.
_SCHEMA_META_RE = re.compile(
    r"\b(?:primary keys?|foreign keys?|join paths?|data catalog|key columns?|primary tables?|"
    r"related tables?|foreign tables?|junction tables?|lookup tables?|database schema|"
    r"database|schema)\b",
    re.I,
)

# A structural dump of primary keys in the model's own words: standalone lines of the form
# "Properties: Id" / "Entities: Id". Bare table names are ordinary English words that must
# stay unredacted in prose, but a line whose ENTIRE content is "<TableName>: Id" can only be
# a schema recitation.
_PK_LINE_RE = re.compile(r"^[ \t]*[A-Z][A-Za-z0-9]+: Id[ \t]*$", re.M)


def redact_real_schema_leaks(text: str) -> str:
    """Strip/substitute any real (non-friendly) schema identifiers, raw SQL, tax IDs, or
    known internal product names from `text` — the one method used everywhere a leak
    might otherwise reach the model or the user, regardless of source (an assistant
    reply, or raw tool-result content). A caught real name is replaced with its friendly
    equivalent where one exists (schema_mapper.real_to_friendly_map()), so the text stays
    informative ("Unknown column 'Type'" rather than "Unknown column '[details
    omitted]'"); names with no friendly form at all (hidden columns like account_id,
    delete_timestamp) fall back to full redaction — there's nothing safe to reveal
    instead.

    Deliberately does NOT touch friendly (PascalCase) catalog labels or generic
    "schema"/"database" narration — see sanitize_user_text() for those, and the module
    docstring for why this function must stay safe to run on structured JSON content
    (those prose-only steps would corrupt a JSON key's exact spelling).
    """
    if not text:
        return text
    out = _SQL_BLOCK_RE.sub(_REDACTED, text)
    out = _SQL_STMT_RE.sub(
        lambda m: _REDACTED if ("`" in m.group(0) or _SQL_KW_RE.search(m.group(0))) else m.group(0), out
    )
    out = _TABLE_COL_RE.sub(_REDACTED, out)
    out = _SCHEMA_WORDS_RE.sub(_replace_real_name, out)
    out = _INTERNAL_PRODUCT_NAMES_RE.sub(_REDACTED, out)
    out = _TAX_ID_RE.sub(_REDACTED, out)
    return out


def sanitize_user_text(text: str) -> str:
    """Strip internal machinery (code, SQL, schema identifiers) from user-facing text —
    everything redact_real_schema_leaks() does, plus prose-only steps that are unsafe to
    apply to structured data.

    Order matters: whole fenced blocks first, then full SQL statements, then narrower
    table.column and bare-identifier references (all via redact_real_schema_leaks()),
    then recited friendly catalog labels (PropertyName, LoanNumber, ...), expanded to
    plain words rather than dropped — the PascalCase format is internal, not the meaning
    (bare "Id" is still redacted). Generic schema-description language ("primary table",
    "key columns", bare "schema"/"database") is stripped outright. Money and normal prose
    pass through untouched.
    """
    if not text:
        return text
    out = redact_real_schema_leaks(text)
    out = _FRIENDLY_SCHEMA_WORDS_RE.sub(lambda m: _FRIENDLY_HUMAN.get(m.group(0), _REDACTED), out)
    out = _PK_LINE_RE.sub(_REDACTED, out)
    out = _SCHEMA_META_RE.sub(_REDACTED, out)
    return out
