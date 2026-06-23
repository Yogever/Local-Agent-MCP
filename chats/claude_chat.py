#!/usr/bin/env python3
"""Claude CLI — uses Anthropic API as the LLM backend."""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import Backend, ToolCall, Usage, init_servers, run_turn
from benchmarking.metrics_logger import MetricsLogger, LOGS_DIR

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096


def get_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.AsyncAnthropic(api_key=api_key)


class ClaudeBackend(Backend):
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def backend_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def build_tools(self, mcp_tools: list) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": dict(t.inputSchema),
            }
            for t in mcp_tools
        ]

    def user_message(self, text: str) -> list:
        return [{"role": "user", "content": text}]

    async def chat(
        self, messages: list, tools: list[dict]
    ) -> tuple[list, list[ToolCall], str | None, Usage]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=messages,
        )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        updated = messages + [{"role": "assistant", "content": response.content}]

        tool_calls = [
            ToolCall(name=b.name, args=dict(b.input), id=b.id)
            for b in response.content
            if b.type == "tool_use"
        ]
        if tool_calls:
            return updated, tool_calls, None, usage

        text = next((b.text for b in response.content if b.type == "text"), "")
        return updated, [], text, usage

    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        return messages + [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": output}
                    for tc, output in results
                ],
            }
        ]


async def main() -> None:
    model = os.environ.get("ANTHROPIC_MODEL", MODEL)
    backend = ClaudeBackend(get_client(), model=model)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = MetricsLogger(LOGS_DIR / "repl_log.jsonl")

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nClaude CLI ({model}) — type 'exit' or Ctrl+C to quit.\n")

        while True:
            try:
                prompt = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                print("Bye.")
                break
            try:
                reply, metrics = await run_turn(
                    backend, tool_to_session, tools, prompt,
                    session_id=session_id,
                    run_label="repl",
                )
                logger.log(metrics)
                print(f"\nClaude: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
