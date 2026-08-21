import argparse
import asyncio

from agent_revamp.agent import Agent
from agent_revamp.pipeline import PreprocessPipeline
from agent_revamp.config import settings
from agent_revamp.core.state import SessionStore


def _print_sessions() -> None:
    store = SessionStore(settings.state_dir)
    sessions = store.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    print("Saved sessions:")
    for s in sessions:
        print(f"  {s['session_id']}  {s['model']}  {s['message_count']} msgs  updated {s['updated_at']}")


def _delete_session(session_id: str) -> None:
    store = SessionStore(settings.state_dir)
    if store.delete(session_id):
        print(f"Deleted session {session_id} and its cached history.")
    else:
        print(f"No saved session found with id {session_id}.")



    
async def main() -> None:
    parser = argparse.ArgumentParser(description="agent-revamp REPL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", metavar="ID", help="resume an existing session")
    group.add_argument("--new", action="store_true", help="start a new session")
    group.add_argument("--list", action="store_true", help="list saved sessions and exit")
    group.add_argument("--delete", metavar="ID", help="delete a session and its cached history, then exit")
    args = parser.parse_args()

    if args.list:
        _print_sessions()
        return

    if args.delete:
        _delete_session(args.delete)
        return

    store = SessionStore(settings.state_dir)
    session_id = None
    if not args.new and not args.session:
        existing = store.list_sessions()
        if existing:
            session_id = existing[0]["session_id"]
            print(f"Resuming latest session {session_id} (use --new for a fresh one).")

    async with Agent(session_id=session_id) as agent:
        print(f"agent-revamp — session {agent.session_id} — connected to penny-mcp. Type 'exit' to quit, "
              f"'/delete' to wipe this session.\n")
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input == "/delete":
                if agent.delete_session():
                    print(f"Deleted session {agent.session_id} and its cached history.")
                else:
                    print(f"No saved session found with id {agent.session_id}.")
                break
            if not user_input:
                continue
            reply = await agent.chat(user_input)
            print(reply, "\n")


async def pipeline_demo(message: str) -> None:
    pipeline = PreprocessPipeline()
    try:
        package = await pipeline.run(message)
        print(
            f"intent : {package.intent.intent} (confidence={package.intent.confidence:.2f})"
        )
        print(f"raw    : {package.raw_query}")
        print("\nskills:")
        for skill in package.skills:
            print(f"  - {skill.name} ({skill.agent})")
        print("\ntools:")
        for tool in package.tools:
            print(f"  - {tool.name} ({tool.agent})")
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
