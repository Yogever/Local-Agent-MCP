"""Scripted benchmark runner — compares with_agent vs without_agent.

Usage (from project root):
    python benchmarking/benchmark.py
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import MCP_SERVERS, Backend, TurnMetrics, init_servers, run_turn
from benchmarking.metrics_logger import MetricsLogger, LOGS_DIR

# ── Prompt list ────────────────────────────────────────────────────────────────
PROMPTS = [
    "Write a Python function that checks if a number is prime.",
    "Write a Python class that implements a stack with push, pop, and peek.",
    "Write a Python script that reads a CSV file and prints the column averages.",
    "Write a function that recursively computes the nth Fibonacci number.",
    "Write a Python decorator that logs how long a function takes to run.",
]

# ── Server configurations ──────────────────────────────────────────────────────
WITH_AGENT_SERVERS = list(MCP_SERVERS.values())
WITHOUT_AGENT_SERVERS = [MCP_SERVERS["filesystem"]]


async def run_condition(
    backend: Backend,
    servers: list,
    label: str,
    prompts: list[str],
    session_id: str,
    logger: MetricsLogger,
) -> list[TurnMetrics]:
    results: list[TurnMetrics] = []
    async with AsyncExitStack() as stack:
        mcp_tools, tool_to_session = await init_servers(stack, servers=servers)
        tools = backend.build_tools(mcp_tools)
        print(f"\n{'─' * 60}")
        print(f"  Run: {label}  |  {len(prompts)} prompts  |  session: {session_id}")
        print(f"{'─' * 60}")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt[:70]}")
            try:
                reply, metrics = await run_turn(
                    backend, tool_to_session, tools, prompt,
                    session_id=session_id,
                    run_label=label,
                )
                logger.log(metrics)
                results.append(metrics)
                print(
                    f"  tokens  in={metrics.total_input_tokens:,} "
                    f"out={metrics.total_output_tokens:,} "
                    f"cached={metrics.total_cached_tokens:,}"
                )
                print(
                    f"  cost    ${metrics.estimated_cost_usd:.6f}  |  "
                    f"latency {metrics.total_latency_ms:.0f}ms "
                    f"(tool {metrics.tool_execution_time_ms:.0f}ms)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    return results


def print_summary(with_results: list[TurnMetrics], without_results: list[TurnMetrics]) -> None:
    def totals(rs: list[TurnMetrics]) -> dict:
        return {
            "input": sum(r.total_input_tokens for r in rs),
            "output": sum(r.total_output_tokens for r in rs),
            "cached": sum(r.total_cached_tokens for r in rs),
            "cost": sum(r.estimated_cost_usd for r in rs),
            "latency": sum(r.total_latency_ms for r in rs) / len(rs) if rs else 0,
        }

    wa = totals(with_results)
    wo = totals(without_results)

    print(f"\n{'═' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<30} {'With agent':>12} {'Without':>12} {'Delta':>12}")
    print(f"  {'─' * 66}")
    print(f"  {'Output tokens':<30} {wa['output']:>12,} {wo['output']:>12,} {wa['output'] - wo['output']:>+12,}")
    print(f"  {'Input tokens':<30} {wa['input']:>12,} {wo['input']:>12,} {wa['input'] - wo['input']:>+12,}")
    print(f"  {'Cached tokens':<30} {wa['cached']:>12,} {wo['cached']:>12,} {wa['cached'] - wo['cached']:>+12,}")
    print(f"  {'Total cost (USD)':<30} {wa['cost']:>12.6f} {wo['cost']:>12.6f} {wa['cost'] - wo['cost']:>+12.6f}")
    print(f"  {'Avg latency (ms)':<30} {wa['latency']:>12.0f} {wo['latency']:>12.0f} {wa['latency'] - wo['latency']:>+12.0f}")
    net_savings = wo['cost'] - wa['cost']
    verdict = "WORTH IT ✓" if net_savings > 0 else "NOT WORTH IT ✗"
    print(f"\n  Net cost savings: ${net_savings:.6f}  →  {verdict}")
    print(f"{'═' * 60}\n")


async def main() -> None:
    # Import the default backend (Groq). Swap this import to benchmark other backends.
    import groq as groq_sdk
    from dotenv import load_dotenv
    load_dotenv()

    from chats.groq_chat import GroqBackend
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY not set.")

    model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    backend = GroqBackend(groq_sdk.AsyncGroq(api_key=api_key), model=model)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{session_id}.jsonl"
    logger = MetricsLogger(log_path)

    print(f"\nBenchmark session: {session_id}")
    print(f"Backend: {backend.backend_name} / {backend.model_name}")
    print(f"Logging to: {log_path}")

    with_results = await run_condition(
        backend, WITH_AGENT_SERVERS, "with_agent", PROMPTS, session_id, logger,
    )
    without_results = await run_condition(
        backend, WITHOUT_AGENT_SERVERS, "without_agent", PROMPTS, session_id, logger,
    )

    print_summary(with_results, without_results)


if __name__ == "__main__":
    asyncio.run(main())
