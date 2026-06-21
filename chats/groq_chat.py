#!/usr/bin/env python3
"""Groq CLI — uses Groq API as the LLM backend."""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import groq
from agent import Backend, ToolCall, init_servers, run_turn

load_dotenv()

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def get_client() -> groq.AsyncGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY environment variable is not set.")
    return groq.AsyncGroq(api_key=api_key)


class GroqBackend(Backend):
    def __init__(self, client: groq.AsyncGroq, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    def build_tools(self, mcp_tools: list) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": dict(t.inputSchema),
                },
            }
            for t in mcp_tools
        ]

    def user_message(self, text: str) -> list:
        return [{"role": "user", "content": text}]

    async def chat(
        self, messages: list, tools: list[dict]
    ) -> tuple[list, list[ToolCall], str | None]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        updated = messages + [msg]

        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                    id=tc.id,
                )
                for tc in msg.tool_calls
            ]
            return updated, tool_calls, None

        return updated, [], msg.content

    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        return messages + [
            {"role": "tool", "content": output, "tool_call_id": tc.id, "name": tc.name}
            for tc, output in results
        ]


async def main() -> None:
    backend = GroqBackend(get_client(), model=MODEL)

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nGroq CLI ({MODEL}) — type 'exit' or Ctrl+C to quit.\n")

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
                print(f"\nGroq: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
