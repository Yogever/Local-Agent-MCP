#!/usr/bin/env python3
"""Gemini CLI — uses Google Gemini API as the LLM backend."""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent import Backend, ToolCall, init_servers, run_turn

load_dotenv()

MODEL = "gemini-2.5-flash"


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


class GeminiBackend(Backend):
    def __init__(self, client: genai.Client) -> None:
        self._client = client

    def _schema(self, s: dict) -> types.Schema:
        t = s.get("type", "object").upper()
        kw: dict = {}
        if desc := s.get("description"):
            kw["description"] = desc
        if t == "OBJECT":
            props = {k: self._schema(v) for k, v in s.get("properties", {}).items()}
            if props:
                kw["properties"] = props
            if req := s.get("required"):
                kw["required"] = req
        elif t == "ARRAY" and (items := s.get("items")):
            kw["items"] = self._schema(items)
        return types.Schema(type=t, **kw)

    def build_tools(self, mcp_tools: list) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=self._schema(dict(t.inputSchema)),
                )
                for t in mcp_tools
            ]
        )

    def user_message(self, text: str) -> list:
        return [types.Content(role="user", parts=[types.Part(text=text)])]

    async def chat(
        self, messages: list, tools: types.Tool
    ) -> tuple[list, list[ToolCall], str | None]:
        response = self._client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=[tools]),
        )
        candidate = response.candidates[0]
        updated = messages + [types.Content(role="model", parts=candidate.content.parts)]

        fc_parts = [p for p in candidate.content.parts if p.function_call]
        if fc_parts:
            tool_calls = [
                ToolCall(name=p.function_call.name, args=dict(p.function_call.args))
                for p in fc_parts
            ]
            return updated, tool_calls, None
        return updated, [], response.text

    def append_tool_results(
        self, messages: list, results: list[tuple[ToolCall, str]]
    ) -> list:
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    name=tc.name, response={"result": output}
                )
            )
            for tc, output in results
        ]
        return messages + [types.Content(role="user", parts=parts)]


async def main() -> None:
    backend = GeminiBackend(get_client())

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack)
        tools = backend.build_tools(mcp_tools)
        print(f"\nGemini CLI ({MODEL}) — type 'exit' or Ctrl+C to quit.\n")

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
                print(f"\nGemini: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
