"""FastAPI entry point for the agent-revamp API (drop-in for realbooks-agents).

Run locally:  uvicorn agent_revamp.api.server:app --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from agent_revamp.api.chat import router as chat_router

app = FastAPI(title="agent-revamp", version="2.0.0")

app.include_router(chat_router)


@app.get("/health-check")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_revamp.api.server:app", host="0.0.0.0", port=8000)
