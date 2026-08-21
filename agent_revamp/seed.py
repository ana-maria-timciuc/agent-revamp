"""Seed Qdrant with the skills and tools catalogs.

Run after starting the Qdrant container:
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
        -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
    python -m agent_revamp.seed

Skills are read from the realbooks-agents skill dirs (penny, dollar_bill, uncle_sam)
unless SKILLS_DIRS overrides them as comma-separated agent_type=path pairs. Tools are
fetched from the penny-mcp server via list_tools() and stored with their OpenAI schema.
Idempotent — re-running upserts by entry id.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastmcp import Client as MCPClient

from agent_revamp.preprocess.catalog import KIND_SKILL, KIND_TOOL, CatalogEntry, QdrantCatalog
from agent_revamp.config import settings
from agent_revamp.preprocess.embeddings import OpenAIEmbeddingService
from agent_revamp.preprocess.tool_sanitizer import sanitize_tool_schema

_DEFAULT_SKILLS_DIRS: dict[str, str] = {
    "penny": "../realbooks-agents/agents/penny/skills",
    "dollar_bill": "../realbooks-agents/agents/dollar_bill/skills",
    "uncle_sam": "../realbooks-agents/agents/uncle_sam/skills",
}


def _resolve_skills_dirs() -> dict[str, str]:
    configured = settings.skills_dirs
    if not configured:
        return _DEFAULT_SKILLS_DIRS
    result: dict[str, str] = {}
    for pair in configured.split(","):
        if "=" not in pair:
            continue
        agent, path = pair.strip().split("=", 1)
        if agent and path:
            result[agent.strip()] = path.strip()
    return result or _DEFAULT_SKILLS_DIRS


def _load_skills() -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    for agent, rel_path in _resolve_skills_dirs().items():
        skills_dir = Path(rel_path).expanduser()
        if not skills_dir.is_absolute():
            # Resolve relative to the repo root (parent of agent_revamp/).
            skills_dir = (Path(__file__).parent.parent / skills_dir).resolve()
        if not skills_dir.is_dir():
            print(f"[seed] WARNING: skills dir not found: {skills_dir}")
            continue
        for path in sorted(skills_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            entries.append(
                CatalogEntry(
                    id=f"skill:{agent}:{path.stem}",
                    kind=KIND_SKILL,
                    name=f"{agent}/{path.stem}",
                    content=content,
                    agent=agent,
                    payload={"file": str(path)},
                )
            )
            print(f"[seed] skill: {agent}/{path.stem}")
    return entries


async def _load_tools() -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    client = MCPClient(settings.penny_mcp_url)
    await client.__aenter__()
    try:
        tools = await client.list_tools()
        for tool in tools:
            schema = sanitize_tool_schema(tool)
            description = schema["function"].get("description") or tool.description or ""
            content = f"Tool {tool.name}. {description}"
            entries.append(
                CatalogEntry(
                    id=f"tool:penny:{tool.name}",
                    kind=KIND_TOOL,
                    name=tool.name,
                    content=content,
                    agent="penny",
                    payload={"openai_schema": schema},
                )
            )
            print(f"[seed] tool: penny/{tool.name}")
    finally:
        await client.__aexit__(None, None, None)
    return entries


async def seed() -> None:
    catalog = QdrantCatalog(embedder=OpenAIEmbeddingService())
    entries = _load_skills() + await _load_tools()
    if not entries:
        print("[seed] Nothing to seed — no skills or tools found.")
        await catalog.close()
        return
    await catalog.upsert(entries)
    await catalog.close()
    print(f"[seed] Done — indexed {len(entries)} entries into Qdrant.")


if __name__ == "__main__":
    asyncio.run(seed())
