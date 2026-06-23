# Local Agent MCP

A benchmarking project that measures whether offloading code generation to a local MCP tool (the coder server) reduces cloud token usage and cost compared to having the LLM generate everything directly.

## Language

### Agentic loop

**Turn**: One user prompt processed through to a final text reply. A turn spans one or more calls to `Backend.chat()`.
_Avoid_: Request, conversation turn, interaction

**Call**: A single invocation of `Backend.chat()`. A turn contains 1..N calls — one per tool-call round-trip plus the final reply call.
_Avoid_: Request, API request, LLM call

**Backend**: An implementation of the `Backend` ABC that adapts a specific LLM provider (Groq, Anthropic, Gemini, Ollama, Llama) to the shared agentic loop.
_Avoid_: Provider, model, client

**Local agent**: The `coder` MCP server. When enabled, the LLM delegates code generation to it rather than producing code in its own output tokens.
_Avoid_: Coder tool, local model, sub-agent

### Metrics

**Usage**: Normalized token counts produced by a single call — `input_tokens`, `output_tokens`, `cached_tokens`. Returned by `Backend.chat()` as a fourth value.
_Avoid_: Token counts, token usage

**CallMetrics**: All measurements for one individual call: Usage, latency, and which tool (if any) was invoked.
_Avoid_: Call stats, call data

**TurnMetrics**: Aggregate measurements for a complete turn. Sums Usage across all its calls; adds turn-level fields: total latency, tool call count, tools used, estimated cost.
_Avoid_: Turn stats, run metrics

### Benchmarking

**Run**: A complete benchmark execution — one backend, one server configuration, one run label, over a fixed prompt list.
_Avoid_: Experiment, test, session

**Run label**: A string tag that identifies the experimental condition of a run. Canonical values: `"with_agent"` and `"without_agent"`.
_Avoid_: Condition, variant, flag

**Session**: A paired benchmark execution — one `with_agent` run and one `without_agent` run that share the same `session_id`. The unit of comparison in the notebook.
_Avoid_: Experiment, comparison, test pair

**Session ID**: A timestamp string (`YYYYMMDD_HHMMSS`) generated once per benchmark session and stamped on every TurnMetrics record in both runs. Enables the notebook to isolate a single session from a multi-session JSONL file.
_Avoid_: Run ID, experiment ID, batch ID

**Prompt list**: The fixed set of user prompts used across both runs of a benchmark. Must be identical between `with_agent` and `without_agent` runs for results to be comparable.
_Avoid_: Test cases, inputs, scenarios

### Analysis

**Net cost savings**: The primary verdict metric. `cost_without_agent − cost_with_agent`, computed after accounting for both output token savings and input token overhead from extra API calls in the agentic loop.
_Avoid_: Token savings, efficiency gain, cost delta

**Input overhead**: The additional input tokens consumed by the `with_agent` run relative to `without_agent`, caused by conversation history growing across multiple API calls in the agentic loop.
_Avoid_: Extra tokens, loop cost, context growth

**Cache hit rate**: `cached_tokens / total_input_tokens` for a run. Measures how effectively the agentic loop reuses cached context across calls.
_Avoid_: Cache ratio, cache efficiency

**Per-prompt ROI ranking**: Prompts sorted by net cost savings descending. Identifies which task types benefit most from the local agent.
_Avoid_: Prompt ranking, savings ranking
