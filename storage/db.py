from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import aiosqlite

log = logging.getLogger(__name__)


SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS channels (
        id                          TEXT PRIMARY KEY,
        display_name                TEXT NOT NULL,
        hashtag                     TEXT NOT NULL,
        mode                        TEXT NOT NULL CHECK (mode IN ('sourced', 'llm_only')),
        research_depth              TEXT NOT NULL DEFAULT 'single'
                                     CHECK (research_depth IN ('single', 'deep')),
        model_writer                TEXT NOT NULL,
        model_researcher            TEXT,
        topic_prompt_active         TEXT NOT NULL,
        research_prompt             TEXT,
        "format"                    TEXT,
        schedule_kind               TEXT NOT NULL,
        schedule_spec               TEXT NOT NULL,
        dedup_window_n              INTEGER NOT NULL DEFAULT 14,
        search_freshness_days       INTEGER,
        search_topic                TEXT,
        images_enabled              INTEGER NOT NULL DEFAULT 0,
        enabled                     INTEGER NOT NULL DEFAULT 1,
        created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages_sent (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id              TEXT NOT NULL REFERENCES channels(id),
        telegram_message_id     INTEGER,
        title                   TEXT NOT NULL,
        body                    TEXT NOT NULL,
        hashtags                TEXT NOT NULL DEFAULT '',
        keywords                TEXT NOT NULL DEFAULT '',
        source_urls             TEXT NOT NULL DEFAULT '[]',
        tokens_in               INTEGER DEFAULT 0,
        tokens_out              INTEGER DEFAULT 0,
        cost_usd                REAL DEFAULT 0,
        created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_channel_time ON messages_sent(channel_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS message_embeddings (
        message_id  INTEGER PRIMARY KEY REFERENCES messages_sent(id) ON DELETE CASCADE,
        channel_id  TEXT NOT NULL,
        model       TEXT NOT NULL,
        dim         INTEGER NOT NULL,
        embedding   TEXT NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_msg_emb_channel_time ON message_embeddings(channel_id, created_at DESC)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_sent_fts USING fts5(
        title, keywords, hashtags, source_urls,
        content='messages_sent', content_rowid='id',
        tokenize='unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_sent_ai AFTER INSERT ON messages_sent BEGIN
        INSERT INTO messages_sent_fts(rowid, title, keywords, hashtags, source_urls)
        VALUES (new.id, new.title, new.keywords, new.hashtags, new.source_urls);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_sent_ad AFTER DELETE ON messages_sent BEGIN
        INSERT INTO messages_sent_fts(messages_sent_fts, rowid, title, keywords, hashtags, source_urls)
        VALUES ('delete', old.id, old.title, old.keywords, old.hashtags, old.source_urls);
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS pending_prompt_updates (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id                  TEXT NOT NULL,
        channel_id                  TEXT NOT NULL REFERENCES channels(id),
        triggered_by_message_id     INTEGER REFERENCES messages_sent(id),
        user_feedback_text          TEXT NOT NULL,
        proposed_prompt             TEXT NOT NULL,
        change_summary              TEXT NOT NULL DEFAULT '',
        proposed_freshness_days     INTEGER,
        proposed_schedule_kind      TEXT,
        proposed_schedule_spec      TEXT,
        proposed_format             TEXT,
        proposed_research_prompt    TEXT,
        status                      TEXT NOT NULL DEFAULT 'pending'
                                     CHECK (status IN ('pending', 'approved', 'cancelled', 'superseded')),
        created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at                 TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pending_session ON pending_prompt_updates(session_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS refinement_sessions (
        session_id      TEXT PRIMARY KEY,
        channel_id      TEXT NOT NULL REFERENCES channels(id),
        message_id      INTEGER REFERENCES messages_sent(id),
        status          TEXT NOT NULL DEFAULT 'open'
                         CHECK (status IN ('open', 'approved', 'cancelled')),
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_ledger (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id      TEXT,
        agent           TEXT NOT NULL,
        model           TEXT NOT NULL,
        tokens_in       INTEGER DEFAULT 0,
        tokens_out      INTEGER DEFAULT 0,
        cost_usd        REAL DEFAULT 0,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_channel_time ON usage_ledger(channel_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_usage_time ON usage_ledger(created_at)",
    """
    CREATE TABLE IF NOT EXISTS source_seen (
        channel_id      TEXT NOT NULL REFERENCES channels(id),
        source_name     TEXT NOT NULL,
        external_id     TEXT NOT NULL,
        first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (channel_id, source_name, external_id)
    )
    """,
]


class Database:
    """Thin async wrapper over aiosqlite."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def execute(self, sql: str, params: tuple | list | dict | None = None) -> aiosqlite.Cursor:
        cur = await self.conn.execute(sql, params or ())
        await self.conn.commit()
        return cur

    async def executemany(self, sql: str, seq: list) -> None:
        await self.conn.executemany(sql, seq)
        await self.conn.commit()

    async def fetchone(self, sql: str, params: tuple | list | dict | None = None) -> Optional[aiosqlite.Row]:
        async with self.conn.execute(sql, params or ()) as cur:
            return await cur.fetchone()

    async def fetchall(self, sql: str, params: tuple | list | dict | None = None) -> list[aiosqlite.Row]:
        async with self.conn.execute(sql, params or ()) as cur:
            return list(await cur.fetchall())


_DB_SINGLETON: Optional[Database] = None


# Additive column migrations for DBs created before a column existed. Each entry is
# (table, column, full column DDL). Applied idempotently via ALTER TABLE ADD COLUMN.
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("pending_prompt_updates", "proposed_freshness_days", "proposed_freshness_days INTEGER"),
    ("pending_prompt_updates", "proposed_schedule_kind", "proposed_schedule_kind TEXT"),
    ("pending_prompt_updates", "proposed_schedule_spec", "proposed_schedule_spec TEXT"),
    ("pending_prompt_updates", "proposed_format", "proposed_format TEXT"),
    ("channels", "research_prompt", "research_prompt TEXT"),
    ("channels", "research_depth", "research_depth TEXT NOT NULL DEFAULT 'single'"),
    ("pending_prompt_updates", "proposed_research_prompt", "proposed_research_prompt TEXT"),
]

# Column renames for DBs created before a column was renamed. Each entry is
# (table, old_column, new_column). Applied only when old exists and new does not.
_RENAME_MIGRATIONS: list[tuple[str, str, str]] = [
    ("channels", "format_partial", "format"),
]


async def _apply_column_migrations(conn: aiosqlite.Connection) -> None:
    for table, old, new in _RENAME_MIGRATIONS:
        async with conn.execute(f"PRAGMA table_info({table})") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if old in cols and new not in cols:
            await conn.execute(
                f'ALTER TABLE {table} RENAME COLUMN {old} TO "{new}"'
            )
            log.info("Migrated %s: renamed column %s -> %s", table, old, new)
    for table, column, ddl in _COLUMN_MIGRATIONS:
        async with conn.execute(f"PRAGMA table_info({table})") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if column not in cols:
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
            log.info("Migrated %s: added column %s", table, column)


async def init_db(path: str) -> Database:
    """Initialise the database singleton and apply schema migrations."""
    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        _DB_SINGLETON = Database(path)
        await _DB_SINGLETON.connect()
        for stmt in SCHEMA_STATEMENTS:
            await _DB_SINGLETON.conn.execute(stmt)
        await _apply_column_migrations(_DB_SINGLETON.conn)
        await _DB_SINGLETON.conn.commit()
        log.info("SQLite initialised at %s", os.path.abspath(path))
    return _DB_SINGLETON


def get_db() -> Database:
    if _DB_SINGLETON is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _DB_SINGLETON
