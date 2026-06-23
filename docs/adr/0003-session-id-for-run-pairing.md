# Session ID for pairing benchmark runs

Each TurnMetrics record carries a `session_id` (format: `YYYYMMDD_HHMMSS`) generated once per benchmark invocation and stamped on every turn in both the `with_agent` and `without_agent` runs. The notebook filters by session ID to isolate a comparison pair from a multi-session JSONL file.

## Considered Options

**Separate files per run** (`benchmark_20260622_with_agent.jsonl`) — simple, no ID needed. Rejected: the notebook needs manual path config updates for every new run, and files proliferate quickly.

**Filter by run label + date** — one file, group by `run_label` and take the most recent date. Rejected: two benchmarks on the same day bleed together with no way to separate them.
