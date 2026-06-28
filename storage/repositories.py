from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from storage.db import Database

log = logging.getLogger(__name__)


# ---------- dataclasses ----------


@dataclass
class ChannelRow:
    id: str
    display_name: str
    hashtag: str
    mode: str
    research_depth: str
    model_writer: str
    model_researcher: Optional[str]
    topic_prompt_active: str
    research_prompt: Optional[str]
    format: Optional[str]
    schedule_kind: str
    schedule_spec: dict
    dedup_window_n: int
    search_freshness_days: Optional[int]
    search_topic: Optional[str]
    images_enabled: bool
    enabled: bool

    @classmethod
    def from_row(cls, row) -> "ChannelRow":
        return cls(
            id=row["id"],
            display_name=row["display_name"],
            hashtag=row["hashtag"],
            mode=row["mode"],
            research_depth=(
                row["research_depth"]
                if "research_depth" in row.keys()
                else "single"
            ),
            model_writer=row["model_writer"],
            model_researcher=row["model_researcher"],
            topic_prompt_active=row["topic_prompt_active"],
            research_prompt=row["research_prompt"],
            format=row["format"],
            schedule_kind=row["schedule_kind"],
            schedule_spec=json.loads(row["schedule_spec"]),
            dedup_window_n=row["dedup_window_n"],
            search_freshness_days=row["search_freshness_days"],
            search_topic=row["search_topic"],
            images_enabled=bool(row["images_enabled"]),
            enabled=bool(row["enabled"]),
        )


@dataclass
class StoredEmbedding:
    message_id: int
    title: str
    created_at: str
    vector: list[float]


@dataclass
class MessageRow:
    id: int
    channel_id: str
    telegram_message_id: Optional[int]
    title: str
    body: str
    hashtags: str
    keywords: str
    source_urls: list[str]
    created_at: str

    @classmethod
    def from_row(cls, row) -> "MessageRow":
        return cls(
            id=row["id"],
            channel_id=row["channel_id"],
            telegram_message_id=row["telegram_message_id"],
            title=row["title"],
            body=row["body"],
            hashtags=row["hashtags"],
            keywords=row["keywords"],
            source_urls=json.loads(row["source_urls"] or "[]"),
            created_at=row["created_at"],
        )


# ---------- repositories ----------


class ChannelRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_from_yaml(self, spec: dict) -> ChannelRow:
        """Insert if new; update non-prompt fields if exists."""
        existing = await self.get(spec["id"])
        schedule_spec_json = json.dumps(spec["schedule"]["spec"])
        if existing is None:
            await self.db.execute(
                """
                INSERT INTO channels (
                    id, display_name, hashtag, mode, research_depth, model_writer, model_researcher,
                    topic_prompt_active, research_prompt, "format", schedule_kind, schedule_spec,
                    dedup_window_n, search_freshness_days, search_topic, images_enabled, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec["id"],
                    spec["display_name"],
                    spec["hashtag"],
                    spec["mode"],
                    spec.get("research_depth", "single"),
                    spec["model_writer"],
                    spec.get("model_researcher"),
                    spec["topic_prompt"],
                    spec.get("research_prompt"),
                    spec.get("format"),
                    spec["schedule"]["kind"],
                    schedule_spec_json,
                    spec.get("dedup_window_n", 14),
                    spec.get("search", {}).get("freshness_days"),
                    spec.get("search", {}).get("topic"),
                    1 if spec.get("images", {}).get("enabled") else 0,
                    1,
                ),
            )
        else:
            await self.db.execute(
                """
                UPDATE channels
                SET display_name=?, hashtag=?, mode=?, research_depth=?, model_writer=?, model_researcher=?,
                    "format"=?, schedule_kind=?, schedule_spec=?, dedup_window_n=?,
                    search_freshness_days=?, search_topic=?, images_enabled=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    spec["display_name"],
                    spec["hashtag"],
                    spec["mode"],
                    spec.get("research_depth", "single"),
                    spec["model_writer"],
                    spec.get("model_researcher"),
                    spec.get("format"),
                    spec["schedule"]["kind"],
                    schedule_spec_json,
                    spec.get("dedup_window_n", 14),
                    spec.get("search", {}).get("freshness_days"),
                    spec.get("search", {}).get("topic"),
                    1 if spec.get("images", {}).get("enabled") else 0,
                    spec["id"],
                ),
            )
        loaded = await self.get(spec["id"])
        assert loaded is not None
        return loaded

    async def get(self, channel_id: str) -> Optional[ChannelRow]:
        row = await self.db.fetchone("SELECT * FROM channels WHERE id=?", (channel_id,))
        return ChannelRow.from_row(row) if row else None

    async def list_enabled(self) -> list[ChannelRow]:
        rows = await self.db.fetchall("SELECT * FROM channels WHERE enabled=1 ORDER BY id")
        return [ChannelRow.from_row(r) for r in rows]

    async def list_all(self) -> list[ChannelRow]:
        rows = await self.db.fetchall("SELECT * FROM channels ORDER BY id")
        return [ChannelRow.from_row(r) for r in rows]

    async def set_topic_prompt(self, channel_id: str, new_prompt: str) -> None:
        await self.db.execute(
            "UPDATE channels SET topic_prompt_active=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (new_prompt, channel_id),
        )

    async def set_research_prompt(self, channel_id: str, new_prompt: Optional[str]) -> None:
        """Set the channel's researcher-only prompt (None = fall back to topic prompt)."""
        await self.db.execute(
            "UPDATE channels SET research_prompt=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (new_prompt, channel_id),
        )

    async def set_research_depth(self, channel_id: str, depth: str) -> None:
        """Set the channel's research depth ('single' = one source, 'deep' = anchor + supporting sources)."""
        await self.db.execute(
            "UPDATE channels SET research_depth=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (depth, channel_id),
        )

    async def set_freshness_days(self, channel_id: str, days: Optional[int]) -> None:
        """Set the channel's content-recency window (max publish-date age, in days)."""
        await self.db.execute(
            "UPDATE channels SET search_freshness_days=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (days, channel_id),
        )

    async def set_format(self, channel_id: str, fmt: Optional[str]) -> None:
        """Set the channel's free-text format instructions (None = use defaults)."""
        await self.db.execute(
            'UPDATE channels SET "format"=?, updated_at=CURRENT_TIMESTAMP WHERE id=?',
            (fmt, channel_id),
        )

    async def set_schedule(
        self, channel_id: str, kind: str, spec: dict
    ) -> None:
        """Set the channel's schedule kind + spec (spec stored as JSON)."""
        await self.db.execute(
            "UPDATE channels SET schedule_kind=?, schedule_spec=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (kind, json.dumps(spec), channel_id),
        )

    async def set_enabled(self, channel_id: str, enabled: bool) -> None:
        await self.db.execute(
            "UPDATE channels SET enabled=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (1 if enabled else 0, channel_id),
        )

    async def delete(self, channel_id: str) -> None:
        await self.db.execute(
            "DELETE FROM source_seen WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute(
            "DELETE FROM usage_ledger WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute(
            "DELETE FROM refinement_sessions WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute(
            "DELETE FROM pending_prompt_updates WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute(
            "DELETE FROM message_embeddings WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute(
            "DELETE FROM messages_sent WHERE channel_id=?", (channel_id,),
        )
        await self.db.execute("DELETE FROM channels WHERE id=?", (channel_id,))


class MessageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert(
        self,
        channel_id: str,
        title: str,
        body: str,
        hashtags: list[str],
        keywords: list[str],
        source_urls: list[str],
        telegram_message_id: Optional[int],
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO messages_sent (
                channel_id, telegram_message_id, title, body, hashtags, keywords,
                source_urls, tokens_in, tokens_out, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                telegram_message_id,
                title,
                body,
                " ".join(hashtags),
                " ".join(keywords),
                json.dumps(source_urls),
                tokens_in,
                tokens_out,
                cost_usd,
            ),
        )
        return cur.lastrowid

    async def update_telegram_id(self, message_id: int, telegram_message_id: int) -> None:
        await self.db.execute(
            "UPDATE messages_sent SET telegram_message_id=? WHERE id=?",
            (telegram_message_id, message_id),
        )

    async def get(self, message_id: int) -> Optional[MessageRow]:
        row = await self.db.fetchone("SELECT * FROM messages_sent WHERE id=?", (message_id,))
        return MessageRow.from_row(row) if row else None

    async def get_by_telegram_id(self, telegram_message_id: int) -> Optional[MessageRow]:
        row = await self.db.fetchone(
            "SELECT * FROM messages_sent WHERE telegram_message_id=?",
            (telegram_message_id,),
        )
        return MessageRow.from_row(row) if row else None

    async def recent_window(self, channel_id: str, n: int) -> list[MessageRow]:
        rows = await self.db.fetchall(
            "SELECT * FROM messages_sent WHERE channel_id=? ORDER BY created_at DESC LIMIT ?",
            (channel_id, n),
        )
        return [MessageRow.from_row(r) for r in rows]

    async def save_embedding(
        self,
        message_id: int,
        channel_id: str,
        model: str,
        vector: list[float],
    ) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO message_embeddings
                (message_id, channel_id, model, dim, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, channel_id, model, len(vector), json.dumps(vector)),
        )

    async def recent_embeddings(
        self,
        channel_id: str,
        days: int,
        limit: int = 500,
    ) -> list[StoredEmbedding]:
        """Stored embeddings for a channel's posts within the last `days` days."""
        rows = await self.db.fetchall(
            """
            SELECT e.message_id AS message_id, e.embedding AS embedding,
                   m.title AS title, m.created_at AS created_at
            FROM message_embeddings e
            JOIN messages_sent m ON m.id = e.message_id
            WHERE e.channel_id = ?
              AND m.created_at >= datetime('now', ?)
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (channel_id, f"-{int(days)} days", limit),
        )
        return [
            StoredEmbedding(
                message_id=r["message_id"],
                title=r["title"],
                created_at=r["created_at"],
                vector=json.loads(r["embedding"]),
            )
            for r in rows
        ]

    async def grep_collisions(
        self,
        channel_id: str,
        title: str,
        keywords: list[str],
        url: str | None = None,
    ) -> list[MessageRow]:
        """FTS5 MATCH over title + keywords + hashtags + source_urls. Returns recent hits."""
        terms: list[str] = []
        for raw in [title, *keywords, url or ""]:
            if not raw:
                continue
            for tok in re.findall(r"[A-Za-z0-9_]{3,}", raw):
                terms.append(tok.lower())
        if not terms:
            return []
        seen = set()
        clean: list[str] = []
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            clean.append(t)
        query = " OR ".join(clean[:20])
        try:
            rows = await self.db.fetchall(
                """
                SELECT m.* FROM messages_sent m
                JOIN messages_sent_fts f ON f.rowid = m.id
                WHERE messages_sent_fts MATCH ? AND m.channel_id = ?
                ORDER BY m.created_at DESC
                LIMIT 10
                """,
                (query, channel_id),
            )
        except Exception as e:
            log.warning("FTS5 query failed (%s); falling back to LIKE", e)
            rows = []
            for tok in clean[:5]:
                rows += await self.db.fetchall(
                    "SELECT * FROM messages_sent WHERE channel_id=? AND lower(title || ' ' || keywords) LIKE ? ORDER BY created_at DESC LIMIT 5",
                    (channel_id, f"%{tok}%"),
                )
        return [MessageRow.from_row(r) for r in rows]


class PendingPromptRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert(
        self,
        session_id: str,
        channel_id: str,
        triggered_by_message_id: Optional[int],
        user_feedback_text: str,
        proposed_prompt: str,
        change_summary: str,
        proposed_freshness_days: Optional[int] = None,
        proposed_schedule_kind: Optional[str] = None,
        proposed_schedule_spec: Optional[dict] = None,
        proposed_format: Optional[str] = None,
        proposed_research_prompt: Optional[str] = None,
    ) -> int:
        await self.db.execute(
            """
            UPDATE pending_prompt_updates
            SET status='superseded', resolved_at=CURRENT_TIMESTAMP
            WHERE session_id=? AND status='pending'
            """,
            (session_id,),
        )
        cur = await self.db.execute(
            """
            INSERT INTO pending_prompt_updates
                (session_id, channel_id, triggered_by_message_id, user_feedback_text,
                 proposed_prompt, change_summary, proposed_freshness_days,
                 proposed_schedule_kind, proposed_schedule_spec, proposed_format,
                 proposed_research_prompt, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                session_id,
                channel_id,
                triggered_by_message_id,
                user_feedback_text,
                proposed_prompt,
                change_summary,
                proposed_freshness_days,
                proposed_schedule_kind,
                json.dumps(proposed_schedule_spec)
                if proposed_schedule_spec is not None
                else None,
                proposed_format,
                proposed_research_prompt,
            ),
        )
        return cur.lastrowid

    async def get_latest_pending(self, session_id: str) -> Optional[dict]:
        row = await self.db.fetchone(
            "SELECT * FROM pending_prompt_updates WHERE session_id=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        return dict(row) if row else None

    async def session_history(self, session_id: str) -> list[dict]:
        rows = await self.db.fetchall(
            "SELECT * FROM pending_prompt_updates WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        )
        return [dict(r) for r in rows]

    async def resolve(self, pending_id: int, status: str) -> None:
        await self.db.execute(
            "UPDATE pending_prompt_updates SET status=?, resolved_at=CURRENT_TIMESTAMP WHERE id=?",
            (status, pending_id),
        )

    async def recent_accepted_for_channel(self, channel_id: str, n: int = 5) -> list[dict]:
        rows = await self.db.fetchall(
            "SELECT * FROM pending_prompt_updates WHERE channel_id=? AND status='approved' ORDER BY resolved_at DESC LIMIT ?",
            (channel_id, n),
        )
        return [dict(r) for r in rows]


class RefinementSessionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def open(self, session_id: str, channel_id: str, message_id: Optional[int]) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO refinement_sessions (session_id, channel_id, message_id, status, updated_at)
            VALUES (?, ?, ?, 'open', CURRENT_TIMESTAMP)
            """,
            (session_id, channel_id, message_id),
        )

    async def get(self, session_id: str) -> Optional[dict]:
        row = await self.db.fetchone(
            "SELECT * FROM refinement_sessions WHERE session_id=?",
            (session_id,),
        )
        return dict(row) if row else None

    async def close(self, session_id: str, status: str) -> None:
        await self.db.execute(
            "UPDATE refinement_sessions SET status=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
            (status, session_id),
        )


class UsageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert(
        self,
        channel_id: Optional[str],
        agent: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO usage_ledger (channel_id, agent, model, tokens_in, tokens_out, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (channel_id, agent, model, tokens_in, tokens_out, cost_usd),
        )

    async def today_total(self, channel_id: Optional[str] = None) -> float:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        if channel_id is None:
            row = await self.db.fetchone(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM usage_ledger WHERE created_at >= ?",
                (since,),
            )
        else:
            row = await self.db.fetchone(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM usage_ledger WHERE channel_id=? AND created_at >= ?",
                (channel_id, since),
            )
        return float(row["s"]) if row else 0.0

    async def today_breakdown(self) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = await self.db.fetchall(
            """
            SELECT channel_id, agent, model,
                   SUM(tokens_in) AS tin, SUM(tokens_out) AS tout, SUM(cost_usd) AS cost
            FROM usage_ledger WHERE created_at >= ?
            GROUP BY channel_id, agent, model
            ORDER BY cost DESC
            """,
            (since,),
        )
        return [dict(r) for r in rows]


class SourceSeenRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def has(self, channel_id: str, source_name: str, external_id: str) -> bool:
        row = await self.db.fetchone(
            "SELECT 1 FROM source_seen WHERE channel_id=? AND source_name=? AND external_id=?",
            (channel_id, source_name, external_id),
        )
        return row is not None

    async def mark(self, channel_id: str, source_name: str, external_id: str) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO source_seen (channel_id, source_name, external_id) VALUES (?, ?, ?)",
            (channel_id, source_name, external_id),
        )
