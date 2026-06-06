from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Approx per-1K-token USD prices for the v1 OpenAI menu.
# Adjust freely; this is a small heuristic, not a billing oracle.
PRICE_TABLE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o":             (0.0025, 0.0100),
    "gpt-4o-mini":        (0.00015, 0.0006),
    "gpt-5.4":            (0.0025, 0.0100),
    "gpt-5.4-mini":       (0.00015, 0.0006),
    "gpt-4.1":            (0.0030, 0.0120),
    "gpt-4.1-mini":       (0.0004, 0.0016),
    "o4-mini":            (0.0011, 0.0044),
    "o3-mini":            (0.0011, 0.0044),
}

DEFAULT_PRICE = (0.0025, 0.0100)


def price_for(model: str) -> tuple[float, float]:
    return PRICE_TABLE_PER_1K.get(model, DEFAULT_PRICE)


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = price_for(model)
    return (tokens_in / 1000.0) * p_in + (tokens_out / 1000.0) * p_out


def usage_from_result(run_result: Any) -> tuple[int, int]:
    """Extract (tokens_in, tokens_out) from a PydanticAI RunResult.

    PydanticAI's usage object exposes request/response token counts. We touch
    several field names defensively because the API has shifted across releases.
    """
    try:
        u = run_result.usage() if callable(getattr(run_result, "usage", None)) else run_result.usage
    except Exception:
        u = None
    if u is None:
        return 0, 0

    def _g(*names: str) -> int:
        for n in names:
            v = getattr(u, n, None)
            if isinstance(v, int):
                return v
        return 0

    tin = _g("request_tokens", "input_tokens", "prompt_tokens")
    tout = _g("response_tokens", "output_tokens", "completion_tokens")
    return tin, tout
