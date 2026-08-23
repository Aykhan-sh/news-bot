from __future__ import annotations

import json
import random
from pathlib import Path

_WORDS_PATH = Path(__file__).resolve().parent.parent / "words.json"

# Only decks already studied are drawn from — words.json ships many more decks
# (Basic III onward, Advanced I-IX) that aren't in scope yet.
_ENABLED_DECKS = {
    "Common Words I",
    "Common Words II",
    "Common Words III",
    "Common Words IV",
    "Common Words V",
    "Common Words VI",
    "Basic I",
    "Basic II",
}

_pool: list[dict] | None = None


def _load_pool() -> list[dict]:
    """Flatten the enabled decks in words.json into a list of {word, definition}.

    Example sentences are dropped here so they never reach a prompt.
    """
    global _pool
    if _pool is None:
        data = json.loads(_WORDS_PATH.read_text(encoding="utf-8"))
        _pool = [
            {"word": w["word"], "definition": w["definition"]}
            for deck in data["decks"]
            if deck["name"] in _ENABLED_DECKS
            for w in deck["words"]
        ]
    return _pool


def sample_words(n: int = 20) -> list[dict]:
    """Return `n` random unique {word, definition} pairs from the word bank."""
    pool = _load_pool()
    return random.sample(pool, min(n, len(pool)))
