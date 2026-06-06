from types import SimpleNamespace

from orchestrator.prompt_builder import render


def _ch():
    return SimpleNamespace(
        id="test",
        display_name="Test",
        hashtag="#test",
        mode="llm_only",
        topic_prompt_active="topic here",
        research_prompt=None,
        search_freshness_days=7,
        search_topic="news",
        format="one bold word, then meaning",
        dedup_window_n=5,
    )


def test_writer_system_prompt_renders():
    text = render(
        "writer.system.j2",
        channel=_ch(),
        window=[{"title": "Alpha", "keywords": "a b"}],
        research_note=None,
    )
    assert "Channel: **Test**" in text
    assert "Alpha" in text
    assert "one bold word, then meaning" in text


def test_researcher_prompt_renders():
    text = render(
        "researcher.j2",
        channel=_ch(),
        window=[],
        fetch_budget=5,
        check_budget=3,
        today="2026-05-30",
        freshness_days=7,
    )
    assert "Researcher" in text
    assert "fetch_budget" not in text  # var was interpolated, not echoed literally
    assert "check_budget" not in text  # var was interpolated, not echoed literally
    assert "web search" in text.lower()
    assert "2026-05-30" in text  # today's date is injected for recency grounding
    assert "check_relevance" in text


def test_researcher_uses_research_prompt_when_set():
    ch = _ch()
    ch.mode = "sourced"
    ch.research_prompt = "watch the official OpenAI and Anthropic blogs"
    text = render(
        "researcher.j2",
        channel=ch,
        window=[],
        fetch_budget=5,
        check_budget=3,
        today="2026-05-30",
        freshness_days=7,
    )
    assert "watch the official OpenAI and Anthropic blogs" in text
    assert "topic here" not in text  # research_prompt overrides the topic fallback


def test_writer_does_not_get_research_prompt():
    ch = _ch()
    ch.mode = "sourced"
    ch.research_prompt = "watch the official OpenAI and Anthropic blogs"
    text = render(
        "writer.system.j2",
        channel=ch,
        window=[],
        research_note=None,
    )
    assert "watch the official OpenAI and Anthropic blogs" not in text
    assert "topic here" in text  # writer still sees the topic prompt


def test_do_not_repeat_empty():
    text = render("shared/do_not_repeat.j2", window=[])
    assert "No recent items" in text
