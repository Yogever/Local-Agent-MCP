"""Shared agentic loop and MCP server management."""

import os
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVERS: dict[str, StdioServerParameters] = {
    "filesystem": StdioServerParameters(
        command="npx",
        args=[
            "@modelcontextprotocol/server-filesystem",
            r"C:\Users\yogev\OneDrive\Documents\Self Learning\Projects\Local Agent MCP\test_folder",
        ],
    ),
    "coder": StdioServerParameters(
        command="python",
        args=["..\mcps\coder_mcp_server.py"],
    ),
}


def get_enabled_servers() -> list[StdioServerParameters]:
    """Return servers filtered by ENABLED_MCPS env var, or all servers if unset."""
    raw = os.environ.get("ENABLED_MCPS", "").strip()
    if not raw:
        return list(MCP_SERVERS.values())
    names = [n.strip() for n in raw.split(",") if n.strip()]
    servers = []
    for name in names:
        if name in MCP_SERVERS:
            servers.append(MCP_SERVERS[name])
        else:
            print(f"  [warn] Unknown MCP server in ENABLED_MCPS: {name!r} (known: {list(MCP_SERVERS)})")
    return servers


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class CallMetrics:
    call_index: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms: float
    tool_names: list[str]  # tools invoked after this call; empty for the final reply call

    def to_dict(self) -> dict:
        return {
            "call_index": self.call_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "tool_names": self.tool_names,
        }


@dataclass
class TurnMetrics:
    session_id: str
    run_label: str
    backend: str
    model: str
    prompt: str
    timestamp: str
    calls: list[CallMetrics]
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    total_latency_ms: float
    tool_execution_time_ms: float
    api_latency_ms: float
    tool_calls_made: int
    tools_used: list[str]
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "run_label": self.run_label,
            "backend": self.backend,
            "model": self.model,
            "prompt": self.prompt,
            "timestamp": self.timestamp,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tool_execution_time_ms": round(self.tool_execution_time_ms, 2),
            "api_latency_ms": round(self.api_latency_ms, 2),
            "tool_calls_made": self.tool_calls_made,
            "tools_used": self.tools_used,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "calls": [c.to_dict() for c in self.calls],
        }


class Backend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier, e.g. 'groq'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string as passed to the API."""

    @abstractmethod
    def build_tools(self, mcp_tools: list) -> object:
        """Convert raw MCP tools to the backend-native tool format."""

    @abstractmethod
    def user_message(self, text: str) -> list:
        """Return an initial messages list containing one user turn."""

    @abstractmethod
    async def chat(
        self, messages: list, tools: object
    ) -> tuple[list, list[ToolCall], str | None, Usage]:
        """
        Call the LLM with the current message history.
        Returns (updated_messages, tool_calls, text, usage).
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
    session_id: str = "",
    run_label: str = "",
) -> tuple[str, TurnMetrics]:
    from pricing import estimate_cost

    messages = backend.user_message(prompt)
    call_metrics_list: list[CallMetrics] = []
    turn_start = time.monotonic()
    total_tool_ms = 0.0
    call_index = 0
    all_tool_names: list[str] = []

    while True:
        call_start = time.monotonic()
        messages, tool_calls, text, usage = await backend.chat(messages, tools)
        call_latency_ms = (time.monotonic() - call_start) * 1000

        call_metrics_list.append(CallMetrics(
            call_index=call_index,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            latency_ms=call_latency_ms,
            tool_names=[tc.name for tc in tool_calls],
        ))
        call_index += 1

        if not tool_calls:
            break

        results: list[tuple[ToolCall, str]] = []
        for tc in tool_calls:
            print(f"  [tool] {tc.name}({tc.args})")
            all_tool_names.append(tc.name)
            t_start = time.monotonic()
            mcp_result = await tool_to_session[tc.name].call_tool(tc.name, tc.args)
            total_tool_ms += (time.monotonic() - t_start) * 1000
            output = "\n".join(
                c.text for c in mcp_result.content if hasattr(c, "text")
            )
            results.append((tc, output))

        messages = backend.append_tool_results(messages, results)

    turn_latency_ms = (time.monotonic() - turn_start) * 1000
    total_input = sum(c.input_tokens for c in call_metrics_list)
    total_output = sum(c.output_tokens for c in call_metrics_list)
    total_cached = sum(c.cached_tokens for c in call_metrics_list)

    metrics = TurnMetrics(
        session_id=session_id,
        run_label=run_label,
        backend=backend.backend_name,
        model=backend.model_name,
        prompt=prompt,
        timestamp=datetime.now(timezone.utc).isoformat(),
        calls=call_metrics_list,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cached_tokens=total_cached,
        total_latency_ms=turn_latency_ms,
        tool_execution_time_ms=total_tool_ms,
        api_latency_ms=turn_latency_ms - total_tool_ms,
        tool_calls_made=len(all_tool_names),
        tools_used=list(dict.fromkeys(all_tool_names)),
        estimated_cost_usd=estimate_cost(backend.model_name, total_input, total_output, total_cached),
    )

    return text, metrics


async def init_servers(
    stack: AsyncExitStack,
    servers: list[StdioServerParameters] | None = None,
) -> tuple[list, dict[str, ClientSession]]:
    """Spawn MCP servers and return (raw_mcp_tools, tool_name→session).

    Pass an explicit server list to override the ENABLED_MCPS env var.
    """
    if servers is None:
        servers = get_enabled_servers()

    tool_to_session: dict[str, ClientSession] = {}
    all_mcp_tools: list = []

    for server_params in servers:
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
