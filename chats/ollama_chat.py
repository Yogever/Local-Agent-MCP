#!/usr/bin/env python3
"""Ollama CLI — uses a local Ollama instance as the LLM backend."""

import asyncio
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
import ollama

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import Backend, ToolCall, Usage, init_servers, run_turn
from benchmarking.metrics_logger import MetricsLogger, LOGS_DIR

MODEL = "deepseek-r1:8b"


class OllamaBackend(Backend):
    def __init__(self, client: ollama.AsyncClient, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def backend_name(self) -> str:
        return "ollama"

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
        response = await self._client.chat(model=self._model, messages=messages, tools=tools)
        msg = response.message

        usage = Usage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            cached_tokens=0,
        )

        updated = messages + [msg]
        if msg.tool_calls:
            tool_calls = [
                ToolCall(name=tc.function.name, args=dict(tc.function.arguments))
                for tc in msg.tool_calls
            ]
            return updated, tool_calls, None, usage
        return updated, [], msg.content, usage

    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        return messages + [
            {"role": "tool", "content": output, "name": tc.name}
            for tc, output in results
        ]


async def main() -> None:
    if MODEL == "your-model-here":
        sys.exit("Error: set the MODEL constant to your Ollama model name.")

    backend = OllamaBackend(ollama.AsyncClient(), model=MODEL)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = MetricsLogger(LOGS_DIR / "repl_log.jsonl")

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nOllama CLI ({MODEL}) — type 'exit' or Ctrl+C to quit.\n")

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
                print(f"\nOllama: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
