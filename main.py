import argparse
import asyncio

from agent_revamp.agent import Agent
from agent_revamp.pipeline import PreprocessPipeline


async def repl() -> None:
    async with Agent() as agent:
        print("agent-revamp — connected to penny-mcp. Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit"):
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="agent-revamp")
    parser.add_argument(
        "mode", nargs="?", default="chat", choices=["chat", "pipeline"], help="run mode"
    )
    parser.add_argument(
        "message", nargs="?", default="", help="message for the pipeline mode"
    )
    args = parser.parse_args()

    if args.mode == "pipeline":
        if not args.message:
            parser.error("pipeline mode requires a message argument")
        await pipeline_demo(args.message)
    else:
        await repl()


if __name__ == "__main__":
    asyncio.run(main())
