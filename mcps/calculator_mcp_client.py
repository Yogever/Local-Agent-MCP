#!/usr/bin/env python3
"""MCP client that connects to the calculator server over stdio."""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command="python",
    args=[".\mcps\calculator_mcp_server.py"],
)


async def main() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Connected to calculator server.")
            print(f"Tools: {[t.name for t in tools.tools]}\n")


if __name__ == "__main__":
    asyncio.run(main())
