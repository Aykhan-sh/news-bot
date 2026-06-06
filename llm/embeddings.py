from __future__ import annotations

import logging
import math
from typing import Sequence

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingService:
    """Thin async wrapper over the OpenAI embeddings endpoint."""

    def __init__(self, api_key: str, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t if (t and t.strip()) else " " for t in texts]
        resp = await self._client.embeddings.create(model=self.model, input=list(cleaned))
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
