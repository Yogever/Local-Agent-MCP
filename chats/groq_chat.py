#!/usr/bin/env python3
"""Groq CLI — uses Groq API as the LLM backend."""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import Backend, ToolCall, Usage, init_servers, run_turn
from benchmarking.metrics_logger import MetricsLogger, LOGS_DIR

load_dotenv()

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_client() -> groq.AsyncGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY environment variable is not set.")
    return groq.AsyncGroq(api_key=api_key)


class GroqBackend(Backend):
    def __init__(self, client: groq.AsyncGroq, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def backend_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

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
    ) -> tuple[list, list[ToolCall], str | None, Usage]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            cached_tokens=(
                getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
                if response.usage and response.usage.prompt_tokens_details
                else 0
            ),
        )

        msg_dict: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        updated = messages + [msg_dict]

        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                    id=tc.id,
                )
                for tc in msg.tool_calls
            ]
            return updated, tool_calls, None, usage

        return updated, [], msg.content, usage

    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        return messages + [
            {"role": "tool", "content": output, "tool_call_id": tc.id, "name": tc.name}
            for tc, output in results
        ]


async def main() -> None:
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    backend = GroqBackend(get_client(), model=model)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = MetricsLogger(LOGS_DIR / "repl_log.jsonl")

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nGroq CLI ({model}) — type 'exit' or Ctrl+C to quit.\n")

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
                print(f"\nGroq: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
