# `Backend.chat()` returns Usage as a fourth value

Token counts are only accessible inside `chat()` — the raw response object from the SDK is not visible anywhere else in the call stack. We extended the return type from `(messages, tool_calls, text)` to `(messages, tool_calls, text, Usage)` so that `run_turn()` can accumulate Usage across the loop without any backend-specific code leaking upward.

## Considered Options

**Injected collector** — pass a `MetricsCollector` into `chat()` and have each backend call it. Rejected: couples the Backend ABC to a specific collector type and scatters recording responsibility across five files with no single source of truth.

**TrackedBackend wrapper** — intercept `chat()` calls in a wrapper class. Rejected: extracting token counts requires knowing which SDK response type you're unwrapping, which re-introduces backend-specific logic in the wrong layer.
