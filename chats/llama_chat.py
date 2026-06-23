#!/usr/bin/env python3
"""Llama 3.2 3B CLI — local Ollama backend."""

import asyncio
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
import ollama

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import init_servers, run_turn
from benchmarking.metrics_logger import MetricsLogger, LOGS_DIR
from chats.ollama_chat import OllamaBackend

MODEL = "llama3.2:3b"


async def main() -> None:
    backend = OllamaBackend(ollama.AsyncClient(), model=MODEL)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = MetricsLogger(LOGS_DIR / "repl_log.jsonl")

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nLlama CLI ({MODEL}) — type 'exit' or Ctrl+C to quit.\n")

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
                print(f"\nLlama: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
