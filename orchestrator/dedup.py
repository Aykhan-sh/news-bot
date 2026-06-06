from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from llm.embeddings import EmbeddingService, cosine_similarity
from storage.repositories import MessageRepo, StoredEmbedding

log = logging.getLogger(__name__)

# Cosine-similarity score above which two posts are treated as the same story.
# We embed title + body, so near-duplicate news (same event, reworded headline)
# lands above this, while genuinely different stories on the same channel sit below.
# Tune here if dedup is too aggressive (raise) or lets duplicates through (lower).
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# How far back the collision check looks. Independent of the freshness window:
# a story we covered two weeks ago is still a duplicate even if "recent" is 3 days.
COLLISION_LOOKBACK_DAYS = 14

_MAX_COMPARE = 500

# How many top matches to record per candidate when score logging is enabled.
_LOG_TOP_K = 5


def build_embed_text(
    title: str,
    keywords: Sequence[str] | str | None = None,
    body: str | None = None,
) -> str:
    """Canonical text we embed for a post/candidate.

    Prefers **title + body** (richer signal, used for posts we actually publish and
    backfill). Falls back to **title + keywords** when no body is available — e.g.
    live web-search candidates in the researcher only carry a title + keywords.
    """
    title = (title or "").strip()
    if body and body.strip():
        return f"{title}\n{body.strip()}".strip()
    if isinstance(keywords, str):
        kw = keywords
    else:
        kw = ", ".join(keywords or [])
    return f"{title}\n{kw}".strip()


@dataclass
class CollisionHit:
    id: int
    title: str
    created_at: str
    score: float


@dataclass
class CollisionReport:
    has_collision: bool
    hits: list[CollisionHit]

    def summary(self) -> str:
        if not self.has_collision:
            return "no collisions"
        return "; ".join(
            f"#{h.id} {h.title!r} (sim={h.score:.2f}) @ {h.created_at}" for h in self.hits
        )


class DedupEngine:
    """Embedding-based near-duplicate detector over a channel's post history."""

    def __init__(
        self,
        messages: MessageRepo,
        embedder: EmbeddingService,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        lookback_days: int = COLLISION_LOOKBACK_DAYS,
        log_candidates: bool = False,
        candidate_log_path: str = "data/dedup_candidates.jsonl",
    ) -> None:
        self.messages = messages
        self.embedder = embedder
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.log_candidates = log_candidates
        self.candidate_log_path = candidate_log_path

    async def sliding_window(self, channel_id: str, n: int):
        return await self.messages.recent_window(channel_id, n)

    async def check(
        self,
        channel_id: str,
        title: str,
        keywords: list[str],
        url: str | None = None,  # kept for call-site compatibility; unused
        days: Optional[int] = None,
        body: str | None = None,
    ) -> CollisionReport:
        """Embed the post (title + body, or title + keywords) and check collisions."""
        vector = await self.embedder.embed(build_embed_text(title, keywords, body))
        return await self.collision_for_vector(
            channel_id, vector, days, candidate_title=title
        )

    async def collision_for_vector(
        self,
        channel_id: str,
        vector: Sequence[float],
        days: Optional[int] = None,
        *,
        preloaded: Optional[list[StoredEmbedding]] = None,
        candidate_title: Optional[str] = None,
    ) -> CollisionReport:
        """Compare a precomputed embedding against the channel's stored embeddings."""
        lookback = days if days is not None else self.lookback_days
        rows = (
            preloaded
            if preloaded is not None
            else await self.messages.recent_embeddings(channel_id, lookback, _MAX_COMPARE)
        )
        scored = [(cosine_similarity(vector, row.vector), row) for row in rows]
        scored.sort(key=lambda s: s[0], reverse=True)
        hits = [
            CollisionHit(
                id=row.message_id,
                title=row.title,
                created_at=row.created_at,
                score=round(score, 3),
            )
            for score, row in scored
            if score >= self.threshold
        ]
        report = CollisionReport(has_collision=bool(hits), hits=hits)
        if self.log_candidates:
            self._log_candidate(channel_id, candidate_title, scored, report)
        if hits:
            log.debug("Collision: best sim=%.3f with #%s", hits[0].score, hits[0].id)
        return report

    def _log_candidate(
        self,
        channel_id: str,
        candidate_title: Optional[str],
        scored: list[tuple[float, StoredEmbedding]],
        report: CollisionReport,
    ) -> None:
        """Append a candidate's top similarity scores to the JSONL audit log.

        Enabled via `dedup.log_candidates`. Lets us review later whether the gate is
        wrongly excluding genuinely-new items (false collisions).
        """
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "channel_id": channel_id,
                "candidate_title": candidate_title,
                "threshold": self.threshold,
                "lookback_days": self.lookback_days,
                "compared": len(scored),
                "has_collision": report.has_collision,
                "top_matches": [
                    {
                        "id": row.message_id,
                        "title": row.title,
                        "score": round(float(score), 4),
                        "collision": score >= self.threshold,
                    }
                    for score, row in scored[:_LOG_TOP_K]
                ],
            }
            path = Path(self.candidate_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # logging must never break the dedup path
            log.warning("Failed to log dedup candidate scores: %s", e)
