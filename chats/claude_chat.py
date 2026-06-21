#!/usr/bin/env python3
"""Claude CLI — uses Anthropic API as the LLM backend."""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import anthropic
from agent import Backend, ToolCall, init_servers, run_turn

load_dotenv()

MODEL = "llama3.2:3b"
MAX_TOKENS = 4096


def get_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.AsyncAnthropic(api_key=api_key)


class ClaudeBackend(Backend):
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

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
    ) -> tuple[list, list[ToolCall], str | None]:
        response = await self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=messages,
        )
        updated = messages + [{"role": "assistant", "content": response.content}]

        tool_calls = [
            ToolCall(name=b.name, args=dict(b.input), id=b.id)
            for b in response.content
            if b.type == "tool_use"
        ]
        if tool_calls:
            return updated, tool_calls, None

        text = next((b.text for b in response.content if b.type == "text"), "")
        return updated, [], text

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
    backend = ClaudeBackend(get_client())

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nClaude CLI ({MODEL}) — type 'exit' or Ctrl+C to quit.\n")

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
                reply = await run_turn(backend, tool_to_session, tools, prompt)
                print(f"\nClaude: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
