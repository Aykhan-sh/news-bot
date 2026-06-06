"""Optional pluggable web-search backend.

v1 uses the LLM provider's **native** web search (OpenAI's `web_search_preview`,
Anthropic's `web_search_*`, …) exposed through PydanticAI's `WebSearchTool`
built-in. The Researcher therefore does NOT call into this module.

The `WebSearch` Protocol below exists only as a future-proofing seam for
providers without a native web-search tool (e.g. local Ollama). It is not wired
into the orchestrator in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol


@dataclass
class SearchHit:
    title: str
    url: str
    domain: str
    snippet: str
    published_at: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "score": self.score,
        }


class WebSearch(Protocol):
    async def search(
        self,
        query: str,
        *,
        days: Optional[int] = None,
        topic: Literal["news", "general"] = "general",
        k: int = 8,
    ) -> list[SearchHit]:
        ...
