"""Pluggable web-search backend.

OpenAI models use the provider's **native** web search (`web_search_preview`)
exposed through PydanticAI's `WebSearchTool` built-in, so they do NOT call into
this module.

Providers without a native web-search tool (Fireworks/GLM and other
OpenAI-compatible endpoints reached via `OpenAIChatModel`) cannot use that
built-in. For them the Researcher exposes a regular `web_search` function tool
backed by a `WebSearch` implementation from this module — `DuckDuckGoSearch`,
which is keyless.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, Optional, Protocol
from urllib.parse import urlparse

log = logging.getLogger(__name__)


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


def _days_to_timelimit(days: Optional[int]) -> Optional[str]:
    """Map a freshness window in days to DuckDuckGo's timelimit code (d/w/m/y)."""
    if not days or days <= 0:
        return None
    if days <= 1:
        return "d"
    if days <= 7:
        return "w"
    if days <= 31:
        return "m"
    return "y"


class DuckDuckGoSearch:
    """Keyless `WebSearch` backed by DuckDuckGo (via the `ddgs` package).

    Used by the Researcher for providers without a native web-search tool. The
    `ddgs` client is synchronous, so each call is run in a worker thread.
    """

    def __init__(self, region: str = "us-en") -> None:
        self.region = region

    async def search(
        self,
        query: str,
        *,
        days: Optional[int] = None,
        topic: Literal["news", "general"] = "general",
        k: int = 8,
    ) -> list[SearchHit]:
        try:
            return await asyncio.to_thread(self._search_sync, query, days, topic, k)
        except Exception as e:  # pragma: no cover - network/runtime failure
            log.warning("DuckDuckGo search failed for %r: %s", query, e)
            return []

    def _search_sync(
        self,
        query: str,
        days: Optional[int],
        topic: Literal["news", "general"],
        k: int,
    ) -> list[SearchHit]:
        from ddgs import DDGS

        timelimit = _days_to_timelimit(days)
        client = DDGS()
        if topic == "news":
            raw = client.news(
                query, region=self.region, timelimit=timelimit, max_results=k
            )
        else:
            raw = client.text(
                query, region=self.region, timelimit=timelimit, max_results=k
            )

        hits: list[SearchHit] = []
        for item in raw:
            url = item.get("url") or item.get("href") or ""
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=item.get("title") or "",
                    url=url,
                    domain=urlparse(url).netloc,
                    snippet=item.get("body") or "",
                    published_at=item.get("date"),
                )
            )
        return hits
