#!/usr/bin/env python3
"""Gemini CLI — connects to multiple MCP servers and exposes their tools to Gemini."""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

MODEL = "gemini-2.5-flash"

# Add or remove MCP servers here — each one is spawned as a subprocess.
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


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def _json_schema_to_gemini(schema: dict) -> types.Schema:
    raw_type = schema.get("type", "object")
    gemini_type = raw_type.upper()
    kwargs: dict = {}

    if desc := schema.get("description"):
        kwargs["description"] = desc

    if gemini_type == "OBJECT":
        props = {
            name: _json_schema_to_gemini(prop)
            for name, prop in schema.get("properties", {}).items()
        }
        if props:
            kwargs["properties"] = props
        if required := schema.get("required"):
            kwargs["required"] = required

    elif gemini_type == "ARRAY":
        if items := schema.get("items"):
            kwargs["items"] = _json_schema_to_gemini(items)

    return types.Schema(type=gemini_type, **kwargs)


async def init_servers(
    stack: AsyncExitStack,
) -> tuple[types.Tool, dict[str, ClientSession]]:
    """Start all MCP servers and return a merged Gemini tool + a tool-name→session map."""
    tool_to_session: dict[str, ClientSession] = {}
    all_declarations: list[types.FunctionDeclaration] = []

    for server_params in MCP_SERVERS:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_result = await session.list_tools()
        for t in tools_result.tools:
            tool_to_session[t.name] = session
            all_declarations.append(
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=_json_schema_to_gemini(dict(t.inputSchema)),
                )
            )
        names = [t.name for t in tools_result.tools]
        print(f"  {server_params.args[0]}: {', '.join(names)}")

    return types.Tool(function_declarations=all_declarations), tool_to_session


async def run_turn(
    client: genai.Client,
    tool_to_session: dict[str, ClientSession],
    gemini_tool: types.Tool,
    prompt: str,
) -> str:
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=prompt)])
    ]

    while True:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(tools=[gemini_tool]),
        )

        candidate = response.candidates[0]
        contents.append(types.Content(role="model", parts=candidate.content.parts))

        function_calls = [p for p in candidate.content.parts if p.function_call]
        if not function_calls:
            return response.text

        result_parts: list[types.Part] = []
        for part in function_calls:
            fc = part.function_call
            print(f"  [tool] {fc.name}({dict(fc.args)})")
            session = tool_to_session[fc.name]
            mcp_result = await session.call_tool(fc.name, dict(fc.args))
            output = "\n".join(
                c.text for c in mcp_result.content if hasattr(c, "text")
            )
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": output},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=result_parts))


async def main() -> None:
    client = get_client()

    print("Starting MCP servers...")
    async with AsyncExitStack() as stack:
        gemini_tool, tool_to_session = await init_servers(stack)
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
                reply = await run_turn(client, tool_to_session, gemini_tool, prompt)
                print(f"\nGemini: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
