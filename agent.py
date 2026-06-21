"""Shared agentic loop and MCP server management."""

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVERS: list[StdioServerParameters] = [
    StdioServerParameters(
        command="npx",
        args=[
            "@modelcontextprotocol/server-filesystem",
            r"C:\Users\yogev\OneDrive\Documents\Self Learning\Projects\Local Agent MCP\test_folder",
        ],
    ),
    StdioServerParameters(
        command="python",
        args=["calculator_mcp_server.py"],
    ),
]


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


class Backend(ABC):
    @abstractmethod
    def build_tools(self, mcp_tools: list) -> object:
        """Convert raw MCP tools to the backend-native tool format."""

    @abstractmethod
    def user_message(self, text: str) -> list:
        """Return an initial messages list containing one user turn."""

    @abstractmethod
    async def chat(
        self, messages: list, tools: object
    ) -> tuple[list, list[ToolCall], str | None]:
        """
        Call the LLM with the current message history.
        Returns (updated_messages, tool_calls, text).
        text is None when tool_calls is non-empty.
        """

    @abstractmethod
    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        """Append tool call results to the message history."""


async def run_turn(
    backend: Backend,
    tool_to_session: dict[str, ClientSession],
    tools: object,
    prompt: str,
) -> str:
    messages = backend.user_message(prompt)

    while True:
        messages, tool_calls, text = await backend.chat(messages, tools)
        if not tool_calls:
            return text

        results: list[tuple[ToolCall, str]] = []
        for tc in tool_calls:
            print(f"  [tool] {tc.name}({tc.args})")
            mcp_result = await tool_to_session[tc.name].call_tool(tc.name, tc.args)
            output = "\n".join(
                c.text for c in mcp_result.content if hasattr(c, "text")
            )
            results.append((tc, output))

        messages = backend.append_tool_results(messages, results)


async def init_servers(
    stack: AsyncExitStack,
) -> tuple[list, dict[str, ClientSession]]:
    """Spawn all MCP_SERVERS and return (raw_mcp_tools, tool_name→session)."""
    tool_to_session: dict[str, ClientSession] = {}
    all_mcp_tools: list = []

    for server_params in MCP_SERVERS:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session: ClientSession = await stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        tools_result = await session.list_tools()
        for t in tools_result.tools:
            tool_to_session[t.name] = session
            all_mcp_tools.append(t)

        names = [t.name for t in tools_result.tools]
        print(f"  {server_params.args[0]}: {', '.join(names)}")

    return all_mcp_tools, tool_to_session
