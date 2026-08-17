import asyncio

from agent_revamp.agent import Agent


async def main() -> None:
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


if __name__ == "__main__":
    asyncio.run(main())
