"""Per-model token pricing. All prices in USD per 1M tokens."""

_PRICING: dict[str, dict[str, float]] = {
    # Groq-hosted models
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11,  "output": 0.34,  "cached": 0.0},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60, "cached": 0.0},
    "llama-3.3-70b-versatile":                      {"input": 0.59,  "output": 0.79,  "cached": 0.0},
    "llama-3.1-8b-instant":                          {"input": 0.05,  "output": 0.08,  "cached": 0.0},
    # Anthropic models
    "claude-haiku-4-5":                              {"input": 0.80,  "output": 4.00,  "cached": 0.08},
    "claude-haiku-4-5-20251001":                     {"input": 0.80,  "output": 4.00,  "cached": 0.08},
    "claude-sonnet-4-6":                             {"input": 3.00,  "output": 15.00, "cached": 0.30},
    "claude-opus-4-8":                               {"input": 15.00, "output": 75.00, "cached": 1.50},
    # Gemini models
    "gemini-2.5-flash":                              {"input": 0.15,  "output": 0.60,  "cached": 0.0},
    "gemini-2.5-pro":                                {"input": 1.25,  "output": 10.00, "cached": 0.0},
    "gemini-2.0-flash":                              {"input": 0.10,  "output": 0.40,  "cached": 0.0},
    # Local / free
    "deepseek-r1:8b":                                {"input": 0.0,   "output": 0.0,   "cached": 0.0},
    "llama3.2:3b":                                   {"input": 0.0,   "output": 0.0,   "cached": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    prices = _PRICING.get(model, {"input": 0.0, "output": 0.0, "cached": 0.0})
    non_cached_input = max(0, input_tokens - cached_tokens)
    return (
        non_cached_input * prices["input"]
        + cached_tokens * prices["cached"]
        + output_tokens * prices["output"]
    ) / 1_000_000
