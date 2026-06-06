import asyncio
import os
import tempfile

import pytest

from storage.db import Database, init_db
from storage.repositories import (
    ChannelRepo,
    MessageRepo,
    PendingPromptRepo,
    UsageRepo,
)


@pytest.fixture
async def db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    # init_db is a singleton; use a fresh Database for isolation.
    db = Database(path)
    await db.connect()
    from storage.db import SCHEMA_STATEMENTS
    for stmt in SCHEMA_STATEMENTS:
        await db.conn.execute(stmt)
    await db.conn.commit()
    try:
        yield db
    finally:
        await db.close()
        os.unlink(path)


@pytest.mark.asyncio
async def test_channel_upsert_and_list(db):
    repo = ChannelRepo(db)
    spec = {
        "id": "test",
        "display_name": "Test",
        "hashtag": "#test",
        "mode": "llm_only",
        "model_writer": "gpt-4o",
        "topic_prompt": "do test things",
        "schedule": {"kind": "interval", "spec": {"hours": 6}},
        "dedup_window_n": 7,
        "format": "one bold word, then meaning",
    }
    row = await repo.upsert_from_yaml(spec)
    assert row.id == "test"
    assert row.dedup_window_n == 7
    assert row.format == "one bold word, then meaning"
    enabled = await repo.list_enabled()
    assert any(c.id == "test" for c in enabled)


@pytest.mark.asyncio
async def test_message_insert_and_window_and_grep(db):
    crepo = ChannelRepo(db)
    await crepo.upsert_from_yaml({
        "id": "ch", "display_name": "ch", "hashtag": "#ch", "mode": "llm_only",
        "model_writer": "gpt-4o-mini", "topic_prompt": "x",
        "schedule": {"kind": "interval", "spec": {"hours": 6}},
    })
    mrepo = MessageRepo(db)
    mid1 = await mrepo.insert(
        channel_id="ch", title="Singular Value Decomposition",
        body="SVD ...", hashtags=["#math"], keywords=["svd", "eigenvalues"],
        source_urls=[], telegram_message_id=None, tokens_in=10, tokens_out=20, cost_usd=0.001,
    )
    assert mid1 > 0
    win = await mrepo.recent_window("ch", 10)
    assert len(win) == 1 and win[0].title.startswith("Singular")

    coll = await mrepo.grep_collisions("ch", "What is SVD?", ["svd"])
    assert any("Singular" in h.title for h in coll)

    none = await mrepo.grep_collisions("ch", "Cooking pasta", ["pasta"])
    assert all("Singular" not in h.title for h in none)


@pytest.mark.asyncio
async def test_usage_aggregation(db):
    crepo = ChannelRepo(db)
    await crepo.upsert_from_yaml({
        "id": "u", "display_name": "u", "hashtag": "#u", "mode": "llm_only",
        "model_writer": "gpt-4o", "topic_prompt": "x",
        "schedule": {"kind": "interval", "spec": {"hours": 6}},
    })
    urepo = UsageRepo(db)
    await urepo.insert("u", "writer", "gpt-4o", 100, 50, 0.0123)
    await urepo.insert("u", "researcher", "gpt-4o-mini", 200, 100, 0.0007)
    total = await urepo.today_total("u")
    assert abs(total - 0.0130) < 1e-6
    breakdown = await urepo.today_breakdown()
    assert len(breakdown) == 2
