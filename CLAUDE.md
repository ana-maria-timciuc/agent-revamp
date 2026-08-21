# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`agent-revamp` is a from-scratch rewrite of the "penny" chat agent currently served by
`realbooks-agents` (see the workspace-root `CLAUDE.md` for how that fits into the platform).
It is a standalone FastAPI + CLI service, **not** one of the standard SAPI/PAPI templates —
it does not depend on `commons` at all (no `commons.db`, no shared health-check router; it
hand-rolls its own minimal `/health-check`).

The service connects to exactly one upstream today: `penny-mcp` (a FastMCP tool server,
sibling repo, reachable at `PENNY_MCP_URL`). Two other MCP servers (`realbooks-dollar-bill-mcp`,
`realbooks-uncle-sam-mcp`) are referenced throughout the code as future connections but are
not wired up — see "Process classes" below.

Comments throughout the codebase say "ported from realbooks-agents/app/agent/..." — when in
doubt about *why* something is shaped a certain way, that sibling repo is the reference
implementation being reproduced (with a different core mechanism, see next section).

## Core architectural idea: the model never sees the real schema

Every other design decision in this repo flows from one goal: the LLM must only ever read
and write **friendly** table/column names (`Transactions.Amount`, `Properties.PropertyName`),
never the real MariaDB schema (`` `transaction`.amount ``, `asset.display_name`) that
`penny-mcp` actually executes against. `agent_revamp/schema_map.json` is the single source of
truth for that friendly↔real mapping (per-table `columns`, `hidden_columns`,
`mandatory_where`, `tool_arg_maps`, `tool_hidden_args`, `tool_description_rewrites`,
`result_tools`). A request flows through the mapping at four distinct points:

1. **`preprocess/schema_mapper.py`** — builds the friendly "data catalog" injected into the
   system prompt (`build_schema_prompt()`), replacing the raw `db://schema` MCP resource.
2. **`preprocess/tool_sanitizer.py`** — rewrites MCP tool schemas into friendly-named OpenAI
   tool defs before the model sees them (`sanitize_tool_schema`), reverses that renaming
   before dispatch (`translate_tool_args`), and injects the real `account_id` server-side
   right before the call (`inject_account_id`) — the model never controls or even sees this
   field.
3. **`preprocess/sql_translator.py`** — the model writes SQL using friendly names;
   `translate_sql()` parses it with `sqlglot`, swaps every identifier for its real column,
   re-aliases `SELECT` output back to friendly names, and injects mandatory `WHERE` clauses
   (account scoping, soft-delete/archive exclusion, split-transaction exclusion) that the
   model must never write itself. Only `SELECT`/`UNION` survives; anything else raises
   `SQLTranslationError`.
4. **`postprocess/result_sanitizer.py`** — renames real DB column keys back to friendly names
   in tool *results* before they re-enter the model's context, drops hidden columns, strips
   control characters, and truncates long strings (defense against prompt injection via
   user-entered row data). Note: `generate_report`'s real SQL is deliberately dropped here
   rather than passed through, unlike the reference implementation it's ported from.

**`postprocess/leak_guard.py`** is the last line of defense, with two entry points sharing
one core:
- `redact_real_schema_leaks()` — the universal method, safe on ANY text (prose or JSON):
  strips fenced code/SQL, real schema words (substituting the friendly equivalent where one
  exists, e.g. `flow_type` → `Type`, via `schema_mapper.real_to_friendly_map()`; falling back
  to opaque redaction only for hidden columns with no friendly form at all, like `account_id`),
  dot-notation table.column references, tax-ID shapes, and known internal product names
  (`MariaDB`). Called from **two** places: `Agent.chat()` on every assistant reply, and
  `Agent._dispatch()` on every tool result right after `result_sanitizer.py` — the latter
  exists because `result_sanitizer.py`'s per-tool-type allow-lists don't cover every shape a
  result can take (a raw DB/driver error surfacing from a `"list"`-type tool like
  `execute_query` bypasses its allow-list entirely; this was a real, confirmed leak until
  fixed on 2026-08-21).
- `sanitize_user_text()` — `redact_real_schema_leaks()` plus prose-only steps unsafe for
  structured data: *expanding* recited friendly catalog labels (`PropertyName` → "Property
  Name") rather than blanking them, and stripping schema-narration language ("primary keys",
  bare "schema"/"database"). Only ever call this on assistant-facing text, never on JSON —
  the friendly-label expansion would corrupt a JSON key's exact spelling.

Both matter because persisted history is replayed as context on every future turn, so a leak
either function misses is effectively permanent. `postprocess/stream_sanitizer.py` is
`sanitize_user_text`'s streaming variant — it buffers partial tokens until a sentence/line
boundary or code-fence close before sanitizing, since a leak can span multiple stream chunks.

When adding a new friendly/real column mapping, edit `schema_map.json` only — every module
above reads it live at call time (no caching), so nothing else needs to change.

**Shared lookups, consolidated 2026-08-21** — don't reintroduce a private copy of these:
`schema_mapper.py::load_schema_map()` is the one canonical `schema_map.json` reader (used by
`sql_translator.py`, `tool_sanitizer.py`, `result_sanitizer.py`, and `schema_mapper.py`
itself — previously each had its own identical private `_load_map()`).
`schema_mapper.py::real_to_friendly_map()` is the one real→friendly identifier map, used by
both `leak_guard.py`'s substitution and `result_sanitizer.py`'s `_build_backstop_map()`
(previously an independently-drifting duplicate of the same ambiguity-drop logic).
`result_sanitizer.py::_build_hidden_set()` and `_sanitize_string()` (control-char/length
hardening on row values) look superficially similar but are *not* duplicates of anything in
`schema_mapper.py`/`leak_guard.py` — deliberately left separate (see those functions'
docstrings for why: different completeness requirements / different concern entirely).

## Process classes and tool scoping

`preprocess/process_class.py` defines four `ProcessClass` values (`penny`, `dollar_bill`,
`uncle_sam`, `transaction_saving`), each with its own `allowed_tools` set, mirroring
`realbooks-agents/app/agent/registry.py::AGENT_REGISTRY`. **Only `penny` is actually wired
up** — `Agent` always connects to `penny_mcp_url` regardless of `process_class`; the other
three registry entries exist so the scoping mechanism is ready the moment those MCP
connections are added. `PROCESS_CLASS` in `.env` controls which allowlist applies.

Enforcement is intentionally double: `filter_tools()` hard-removes disallowed tools right
after `list_tools()` in `Agent.__aenter__`; `validate_tool_scope()` re-checks every tool
about to be shown to the model on every turn and raises `ToolScopeError` (refusing the turn)
if anything slipped through — specifically a guard against the Qdrant-based tool retrieval
(the preprocess pipeline, see below) ever surfacing something out of scope, e.g. from a
shared `QDRANT_TOOLS_COLLECTION` carrying entries from a different process class.

## Preprocess pipeline: intent classifier -> RAG -> reranker

`agent_revamp/preprocess/` holds one consolidated pipeline (`pipeline.py::PreprocessPipeline`)
that `Agent.chat()` calls once per turn (`Agent._run_preprocessing`), matching the architecture
diagram's "Preprocess" box: classify intent, retrieve candidate skills+tools from Qdrant,
rerank, retry on a failed rerank, and hand the result (`pipeline.py::ContextPackage`) back to
`Agent`. This used to be two separate, non-interoperating implementations (a structured-but-
unwired scaffold vs. a simpler live path with no intent/reranking step) that a git merge
unioned without reconciling — they've since been merged into one; if you see a reference to
`core/vector.py`, a top-level `agent_revamp/catalog.py`/`pipeline.py`/`intent.py`/
`reranker.py`/`embeddings.py`, or `agent.py::_retrieve_tools`/`_retrieve_skills` anywhere
(old commits, notes, muscle memory), it's stale — those were deleted/moved into `preprocess/`.

- **`preprocess/catalog.py`** (`QdrantCatalog`) is the RAG box: two Qdrant collections
  (`QDRANT_SKILLS_COLLECTION`/`QDRANT_TOOLS_COLLECTION`), pluggable `EmbeddingService`
  (`preprocess/embeddings.py`) and `Reranker` (`preprocess/reranker.py`). Every public method
  (`upsert`/`search`/`close`) degrades gracefully — a downed Qdrant never raises, callers get
  an empty result and the agent falls back to "no RAG this turn." `search()` owns the
  diagram's fail->retry loop internally (`_search_kind`): on `RerankOutcome.passed=False` it
  widens `top_k` by `RERANK_RETRY_TOP_K_MULTIPLIER` and re-queries, up to `RERANK_MAX_RETRIES`
  times, then returns the best-effort last attempt. `upsert()` prunes orphaned points (renamed/
  removed skills or tools) scoped to each entry's own `agent` field — this matters because
  `dollar_bill`/`uncle_sam` share the same collections as `penny`; an unscoped prune would let
  one process class's reindex delete another's entries.
- **`preprocess/reranker.py`** (`Reranker` ABC, `RankedHit`, `RerankOutcome`) is deliberately
  left abstract — `ScoreReranker` (re-sort by vector score, "pass" whenever there's a hit) is
  a placeholder default, not a real reranking strategy. A concrete cross-encoder/multi-vector/
  LLM-judge/custom implementation is still being chosen; only swap `ScoreReranker` for a real
  one, don't touch the ABC's shape unless the pass/fail contract itself needs to change.
- **`preprocess/intent.py`** (`IntentClassifier` ABC, `LLMIntentClassifier`) is similarly
  left abstract for the same reason — `LLMIntentClassifier` (one OpenAI call against
  `INTENT_TAXONOMY`) is today's only implementation. It takes its OpenAI client via
  constructor injection (not a module-level singleton), so `preprocess/` has zero dependency
  on `agent.py` — keep it that way to avoid a circular import.
- **`preprocess/embeddings.py`** also has an in-progress `GemmaEmbeddingService` stub
  (`embed_texts`/`embed_text` raise `NotImplementedError`) — not wired up as the default
  anywhere; finish it and pass it explicitly to `QdrantCatalog(embedder=...)` when ready.
- **`main.py --preprocess MESSAGE`** is the debug entrypoint for this whole box: it runs
  `PreprocessPipeline.run()` (real intent classification + real Qdrant retrieval) and prints
  intent/confidence, reranker pass/fail per kind, and retrieved skill/tool names — but never
  touches `Agent`/the chat-completion loop, so it's a cheap way to inspect retrieval behavior
  without burning a full agent turn.
- **`agent_revamp/seed.py`** (`python -m agent_revamp.seed`) batch-populates the same two
  Qdrant collections ahead of time from sibling-repo skill directories (`SKILLS_DIRS`) plus a
  live `penny-mcp` tool list — useful for pre-warming Qdrant, but `Agent.__aenter__` also
  indexes live on every startup regardless, so seeding isn't required for the agent to work.

`Agent.chat()`'s tool-candidate handling has three distinct outcomes, worth knowing before
touching that code: no pipeline (Qdrant down) -> full tool set; pipeline ran but retrieved
zero tool candidates -> full tool set; pipeline retrieved candidates but none survive the
`self._tool_schemas` process-class membership check -> **empty** tool list, not the full set
(a retrieved name outside the allowed set must never silently upgrade to "full access").
`validate_tool_scope()`/`filter_tools()` (see "Process classes" above) are unrelated to and
unaffected by any of this — they're a separate, still-untouched enforcement layer.

Retrieved intent (when `is_confident`, i.e. `confidence >= 0.6`) and retrieved skill content
are folded into the prompt as extra system-role messages (`Agent.chat()`, right before the
`validate_tool_scope` call) — retrieved tools go into the OpenAI `tools=` kwarg as structured
data, never as prose.

## Request lifecycle (API path)

`api/server.py` mounts `api/chat.py`'s router. Endpoints mirror `realbooks-agents/app/api/chat.py`
field-for-field (`api/models.py`) so `realbooks-reporting-papi`'s proxy to `agents_api` needs no
changes to point here instead:

- `POST /agent/{slug}/chat` and `/chat/stream` (plus legacy unslugged variants that hardcode
  `penny`) — `dollar-bill`/`uncle-sam` slugs resolve but 404 since only `penny` is registered
  in `_REGISTRY`; `/agent/{slug}/approve` always 404s (penny's tools are read-only, so
  `approval_required` is never emitted).
- Auth (`api/auth.py`) is always-on, fail-closed: `Authorization: Bearer <token>` must be a
  Cognito access token verified via JWKS (`COGNITO_JWKS_URL`); `account_id`/`role_key`/
  `permissions` are then looked up from the users SAPI (`USERS_DOMAIN`) by the verified `sub`
  claim and cached 10s — identity is never trusted from the request body.
- Streaming (`api/sse.py`) relays a producer's queued events as SSE with 15s keep-alives;
  `Agent.chat(..., event_sink=...)` emits `stage`/`tools`/`token` events, sanitized per-chunk
  via `stream_sanitizer.emit_sanitized`.
- After a successful `generate_report` tool call, the turn short-circuits: no second LLM
  round-trip happens, a summary is built locally from the sanitized report payload
  (`agent.py::_report_summary_message`) — mirrors the original harness's report handling.

The CLI (`main.py`) exercises the same `Agent` class directly (no HTTP/auth layer) with a
`SessionStore`-backed REPL; `--list`/`--delete`/`--session`/`--new` manage JSON session files
in `STATE_DIR` (default `state/`).

## State manager (`core/state.py::SessionStore`)

Matches the diagram's "State manager" box (dashed lines fanning out to every other box):
alongside `messages`, each session file's `turns` array is a per-response trace, one entry
per `Agent.chat()` call, built by `agent.py::Agent._record_turn` and appended to at every one
of `chat()`'s four return points (tool-scope block, final answer, report short-circuit, max
iterations exhausted). Each entry captures:

- **Time**: `started_at`/`finished_at`/`duration_ms` (wall-clock via `time.monotonic()`).
- **Tokens**: `tokens_used` (this turn only) and `tokens_total` (cumulative, `Agent.tokens_used`).
- **Variables**: the context that shaped the turn — `process_class`, `account_id`, `model`,
  `tools_offered`, and (when the intent classifier was confident) `intent`/`intent_confidence`
  and `skills_retrieved`.
- **Stages**: one status string per box the diagram draws — `preprocess` (ran with
  skill/tool/reranker counts, or skipped if Qdrant/indexing was down),
  `tool_scope_validation` (passed/blocked), `tool_dispatch` (call count / error count),
  `postprocess` (which sanitizer ran), `report_short_circuit` (yes/no).
- **Tools**: `[{"name": ..., "status": "ok"|"error"}, ...]` for every tool call dispatched
  that turn — `"error"` is a best-effort check (`agent.py::_is_error_result`) for
  `call_tool_safe`'s `{"error": ...}` shape; it only labels the trace, never affects control
  flow.

Bounded at `_MAX_TRACKED_TURNS` (200) per session, same "don't grow a long-lived session
file forever" reasoning as `MAX_HISTORY_MESSAGES` for messages. `SessionStore` itself treats
`turns` as an opaque list it persists/reloads alongside `messages` — it has no knowledge of
the shape above.

## Common commands

```bash
# Local setup (from this directory)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # then fill in OPENAI_KEY, COGNITO_JWKS_URL, USERS_DOMAIN, etc.

# Qdrant (required for dynamic tool/skill retrieval; agent degrades gracefully without it)
docker-compose up -d      # brings up qdrant on :6333/:6334

# CLI REPL
python main.py                    # resumes latest session, or starts fresh if none
python main.py --new              # force a new session
python main.py --list             # list saved sessions
python main.py --delete <id>      # delete a session
python main.py --preprocess "some question"   # debug intent+RAG+rerank only, no agent turn

# API server
uvicorn agent_revamp.api.server:app --reload --port 8000
curl http://localhost:8000/health-check

# Tests (added 2026-08-21 — pytest only, no pytest-asyncio: async code is driven with
# plain asyncio.run() inside sync `def test_...():` functions, mirroring the convention
# realbooks-agents' own test suite uses)
pip install -r requirements-dev.txt
pytest
```

There is no lint config (`ruff.toml`) or `Makefile` in this repo yet — don't assume `ruff`
targets exist here the way they do in the platform's other Python services. `tests/` covers
the ported/security-sensitive modules (`sql_translator`, `tool_sanitizer`, `result_sanitizer`,
`leak_guard`, `stream_sanitizer`, `process_class`, `core/mcp_tools`) and the merged
preprocessing pipeline/`Agent.chat()` integration — not the API layer (`api/*`) or CLI REPL
loop yet.

Ground-truthing that ported suite against this repo's real behavior surfaced, and all as of
2026-08-21 fixed, three `sql_translator.py` bugs:
- An `ORDER BY <select-list-alias>` reference and a subquery-alias-qualified column both
  used to raise `SQLTranslationError` instead of resolving; `translate_sql()` now tracks
  `select_aliases`/`derived_aliases` in `_translate_column` to recognize both as legitimately
  not a catalog column, rather than erroring.
- `_add_mandatory_where` used to always attach mandatory `WHERE` clauses (account scoping,
  soft-delete/split-transaction exclusion) to the top-level query, even when the table they
  reference is nested inside a subquery — producing invalid SQL that referenced a table out
  of scope. Fixed by tracking each table alias's enclosing `exp.Select` (`alias_scopes`, set
  in `_translate_table` via `node.find_ancestor(exp.Select)`) and attaching each table's
  mandatory filters to its own scope instead of always the outermost query. A top-level query
  with no subqueries is unaffected (every alias's scope is just the query itself).

Also fixed the same day: `skills/transactions.md` used to leak real snake_case column names
(rewritten to friendly names); `leak_guard.py`'s `_INTERNAL_PRODUCT_NAMES_RE` now catches
"MariaDB" even outside a full SQL statement (a small hand-maintained supplement to the
schema-derived `_SCHEMA_WORDS_RE`, since a database engine name can never be derived from
`schema_map.json`).

## Key settings (`agent_revamp/config.py`, from `.env`)

- `PENNY_MCP_URL`, `PROCESS_CLASS` — which MCP server and tool allowlist this instance runs as.
- `DEFAULT_ACCOUNT_ID` — pinned account scope used by the CLI (no per-user auth there); the
  API path always uses the authenticated caller's real `account_id` instead.
- `MAX_TOOL_ITERATIONS`, `TOOL_CALL_TIMEOUT_SECONDS` — bound the agent's tool-call loop and
  per-call timeout (`core/mcp_tools.py::call_tool_safe` turns a timeout/exception into a JSON
  `{"error": ...}` tool result rather than raising, so one bad call can't hang the turn).
- `MAX_HISTORY_MESSAGES`, `MAX_MESSAGE_CHARS` — bound persisted-history replay per turn
  (`Agent._history_for_llm`); in-turn messages are never trimmed.
- `QDRANT_*`, `SKILLS_DIR` — the preprocess pipeline's collection names/top-k and where
  `skills/*.md` files are loaded from (one file = one skill, indexed by filename stem).
- `RERANK_MAX_RETRIES` (default 2), `RERANK_RETRY_TOP_K_MULTIPLIER` (default 2) — bound the
  reranker fail->retry loop in `preprocess/catalog.py::QdrantCatalog._search_kind`.
- `INTENT_TAXONOMY` — the fixed label set `preprocess/intent.py::LLMIntentClassifier`
  classifies into.
- `SKILLS_DIRS` — only consumed by `agent_revamp/seed.py`'s standalone batch-seeding script,
  not by `Agent`'s live indexing (which always uses the single flat `SKILLS_DIR`).
