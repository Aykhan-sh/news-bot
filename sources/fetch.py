from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; news-bot/1.0; +https://github.com/) "
        "PythonAiohttp"
    ),
}


@dataclass
class ArticleExcerpt:
    url: str
    title: Optional[str]
    byline: Optional[str]
    published_at: Optional[str]
    text: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "byline": self.byline,
            "published_at": self.published_at,
            "text": self.text,
        }


async def fetch_url_excerpt(url: str, max_chars: int = 6000, timeout: int = 20) -> ArticleExcerpt:
    """Fetch a URL and return a trafilatura-cleaned text excerpt."""
    try:
        async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as sess:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                html = await resp.text(errors="ignore")
    except Exception as e:
        log.warning("Fetch failed for %s: %s", url, e)
        return ArticleExcerpt(url=url, title=None, byline=None, published_at=None, text="")

    text, title, author, date = await asyncio.to_thread(_extract_with_trafilatura, html, url)
    excerpt = (text or "")[:max_chars]
    return ArticleExcerpt(url=url, title=title, byline=author, published_at=date, text=excerpt)


def _extract_with_trafilatura(html: str, url: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        import trafilatura
        from trafilatura.metadata import extract_metadata
    except Exception as e:
        log.debug("trafilatura unavailable (%s); returning raw text", e)
        return _strip_html(html), None, None, None

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    ) or ""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    try:
        meta = extract_metadata(html)
        if meta is not None:
            title = getattr(meta, "title", None)
            author = getattr(meta, "author", None)
            date = getattr(meta, "date", None)
    except Exception:
        pass
    return text, title, author, date


def _strip_html(html: str) -> str:
    import re

    no_tags = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", no_tags).strip()
