#!/usr/bin/env python3
"""MCP server exposing code-generation tools backed by a local Ollama model."""
import types
import ollama
from mcp.server.fastmcp import FastMCP
import json
mcp = FastMCP("coder")

@mcp.tool()
def write_greet_function(greeting: str) -> types.FunctionType:
    """Creates a function meant to print a greeting and a name """
    def greet_function(name: str):
        print(f'{greeting} {name}')

    return greet_function.__code__


@mcp.tool()
def generate_code(task: str, language: str = "python") -> str:
    """Generate code using a local qwen2.5-coder:7b model.

    Args:
        task: Plain-text instruction for what the code should do, e.g. "a function that adds two numbers".
        language: Target programming language (default: python).
    """
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a code generator. Output only {language} code. "
                    "No explanations, no markdown fences, no comments unless asked."
                ),
            },
            {"role": "user", "content": task},
        ],
    )
    return response.message.content


if __name__ == "__main__":
    mcp.run(transport="stdio")
