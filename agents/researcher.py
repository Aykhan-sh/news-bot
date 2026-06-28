from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as date_parser
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

try:
    from pydantic_ai.native_tools import WebSearchTool
except ImportError:  # pragma: no cover - older pydantic-ai layout
    try:
        from pydantic_ai.builtin_tools import WebSearchTool
    except ImportError:
        from pydantic_ai.tools import WebSearchTool  # type: ignore[no-redef]

from agents.deps import Candidate, ResearcherDeps
from orchestrator.dedup import build_embed_text
from orchestrator.prompt_builder import render
from sources.fetch import fetch_url_excerpt

log = logging.getLogger(__name__)

DEFAULT_RELEVANCE_DAYS = 7


class ResearcherOutput(BaseModel):
    picked_id: Optional[str] = Field(
        default=None,
        description=(
            "The id (e.g. 's1') of the fetched source to publish, exactly as the "
            "`fetch_url` tool returned it. It MUST be a source you fetched this "
            "tick (so the writer has its full text) and that came back "
            "`is_relevant` from `check_relevance`. Null when nothing fits."
        ),
    )
    picked_title: Optional[str] = Field(
        default=None,
        description=(
            "A 3-5 word title of the source you picked in `picked_id`, copied from "
            "that fetched source. Used only for logging/debugging so the chosen id "
            "is easy to verify. Null when `picked_id` is null."
        ),
    )
    supporting_ids: list[str] = Field(
        default_factory=list,
        description=(
            "DEEP-RESEARCH channels only, and normally left empty. Supporting sources "
            "are committed by calling the `gather_supporting_sources` tool, not via "
            "this field — the tool's accepted set is what reaches the writer. Only "
            "list ids here as a fallback if you could not call that tool; each must be "
            "a source you fetched this tick. Never include the anchor id."
        ),
    )


async def _system_prompt(ctx: RunContext[ResearcherDeps]) -> str:
    return render(
        "researcher.j2",
        channel=ctx.deps.channel,
        fetch_budget=ctx.deps.fetch_budget,
        check_budget=ctx.deps.check_budget,
        research_depth=ctx.deps.channel.research_depth,
        deep_max_sources=ctx.deps.deep_max_sources,
        today=datetime.now(timezone.utc).date().isoformat(),
        freshness_days=ctx.deps.channel.search_freshness_days or DEFAULT_RELEVANCE_DAYS,
    )


async def web_search(
    ctx: RunContext[ResearcherDeps],
    query: str,
    max_results: int = 8,
) -> list[dict]:
    """Search the live web for `query` and return ranked hits.

    Use this to discover candidate articles before reading them. Each hit has a
    `url`, `title`, `snippet`, and (when known) `published_at`. Pick the most
    promising urls and pass them to `fetch_url` to read the full text. The search
    is scoped to this channel's freshness window and topic mode automatically.
    """
    channel = ctx.deps.channel
    topic = "news" if (channel.search_topic or "general") == "news" else "general"
    hits = await ctx.deps.web_search.search(
        query,
        days=channel.search_freshness_days or DEFAULT_RELEVANCE_DAYS,
        topic=topic,  # type: ignore[arg-type]
        k=max_results,
    )
    return [
        {
            "url": h.url,
            "title": h.title,
            "snippet": h.snippet,
            "published_at": h.published_at,
        }
        for h in hits
    ]


async def fetch_url(
    ctx: RunContext[ResearcherDeps],
    url: str,
    max_chars: int = 6000,
) -> dict:
    """Fetch and clean an article excerpt, registering it as a pickable candidate.

    Call this after the native web-search tool surfaces a URL you want to read
    in full. The article is stored internally under a short id (e.g. `s1`) and
    that id is returned to you. **You publish by returning that id as
    `picked_id`** — only fetched candidates can be picked. Fetch any candidate
    you might choose before committing.
    """
    if ctx.deps.fetch_calls_made >= ctx.deps.fetch_budget:
        return {"error": f"fetch budget exhausted ({ctx.deps.fetch_budget})"}
    ctx.deps.fetch_calls_made += 1
    excerpt = await fetch_url_excerpt(url, max_chars=max_chars)
    source_id = f"s{len(ctx.deps.fetched) + 1}"
    candidate = Candidate(
        id=source_id,
        url=excerpt.url,
        title=excerpt.title,
        body=excerpt.text,
        published_at=excerpt.published_at,
        byline=excerpt.byline,
    )
    ctx.deps.fetched[source_id] = candidate
    return {
        "id": source_id,
        "url": candidate.url,
        "title": candidate.title or "",
        "published_at": candidate.published_at,
        "body": candidate.body,
    }


def _date_verdict(
    published_at: Optional[str], window_days: int, now: datetime
) -> dict:
    """Deterministic freshness check for a single item's publish date.

    Returns `within_window` plus a `date_verdict` of
    `fresh` / `stale` / `future_date` / `unknown_date` / `unparseable_date`.
    Used by the `check_relevance` tool to guide the model.
    """
    if not published_at:
        return {
            "published_at": None,
            "age_days": None,
            "within_window": False,
            "date_verdict": "unknown_date",
        }
    try:
        parsed = date_parser.parse(published_at)
    except (ValueError, OverflowError, TypeError):
        return {
            "published_at": published_at,
            "age_days": None,
            "within_window": False,
            "date_verdict": "unparseable_date",
        }
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = (now - parsed).total_seconds() / 86400.0
    within = 0 <= age_days <= window_days
    return {
        "published_at": parsed.date().isoformat(),
        "age_days": round(age_days, 1),
        "within_window": within,
        "date_verdict": (
            "fresh" if within else ("future_date" if age_days < 0 else "stale")
        ),
    }


async def check_relevance(
    ctx: RunContext[ResearcherDeps],
    freshness_days: Optional[int] = None,
) -> dict:
    """Relevance gate for all candidates you have fetched so far — call it after fetching.

    Automatically processes every candidate fetched via `fetch_url` that has not
    been checked yet. For each one it runs:

    1. **Date relevance** — parses `published_at` and compares it to today.
       Anything older than the recency window is `stale` and must NOT be posted.
    2. **Duplicate / collision** — embeds the candidate (title + body) and
       compares by cosine similarity against this channel's recent posts. A high
       score means we already covered that story.

    A candidate is relevant only when it is BOTH fresh AND not a duplicate.

    The response lists every relevant candidate with its `id`, `url`, and `title`
    so you can decide which one to pick as `picked_id`. If none are relevant,
    fetch more URLs and call this again — but you only get a limited number of
    rounds (see `check_budget`). When the budget is spent the call returns an
    `error`; at that point pick the best item that was relevant in an earlier
    round, or return `picked_id: null`.

    Args:
        freshness_days: optional tighter window (e.g. 3 if the topic prompt says
            "last 3 days"). If omitted, the agent's default window is used.
    """
    window_days = (
        freshness_days
        if freshness_days is not None
        else (ctx.deps.channel.search_freshness_days or DEFAULT_RELEVANCE_DAYS)
    )
    now = datetime.now(timezone.utc)

    unchecked = [c for c in ctx.deps.fetched.values() if not c.was_checked]

    if not unchecked:
        already_relevant = [
            {"id": c.id, "url": c.url, "title": c.title or ""}
            for c in ctx.deps.fetched.values()
            if c.is_relevant
        ]
        return {
            "message": "No new candidates to check.",
            "relevant": already_relevant,
            "recommended_id": already_relevant[0]["id"] if already_relevant else None,
        }

    if ctx.deps.check_calls_made >= ctx.deps.check_budget:
        return {
            "error": (
                f"collision-check budget exhausted ({ctx.deps.check_budget}). "
                "Stop searching: pick the best candidate that already came back "
                "relevant in a previous check, or return picked_id: null."
            ),
            "checks_used": ctx.deps.check_calls_made,
            "check_budget": ctx.deps.check_budget,
        }
    ctx.deps.check_calls_made += 1

    vectors = await ctx.deps.dedup.embedder.embed_many(
        [build_embed_text(c.title or "", c.keywords, c.body) for c in unchecked]
    )
    preloaded = await ctx.deps.messages.recent_embeddings(
        ctx.deps.channel.id, ctx.deps.dedup.lookback_days, 500
    )

    for i, cand in enumerate(unchecked):
        ctx.deps.fetched_vectors[cand.id] = vectors[i]
        date_info = _date_verdict(cand.published_at, window_days, now)
        report = await ctx.deps.dedup.collision_for_vector(
            ctx.deps.channel.id,
            vectors[i],
            preloaded=preloaded,
            candidate_title=cand.title,
        )
        cand.was_checked = True
        cand.is_relevant = bool(date_info["within_window"]) and not report.has_collision

    relevant = [
        {"id": c.id, "url": c.url, "title": c.title or ""}
        for c in ctx.deps.fetched.values()
        if c.is_relevant
    ]
    recommended_id = relevant[0]["id"] if relevant else None

    if relevant:
        label = ", ".join(f"{r['id']} — {r['title']}" for r in relevant)
        message = f"The following {len(relevant)} candidate(s) are relevant: {label}"
    else:
        message = "No relevant candidates found among the checked items."

    return {
        "freshness_days": window_days,
        "checked": len(unchecked),
        "relevant": relevant,
        "recommended_id": recommended_id,
        "message": message,
    }


async def choose_anchor(
    ctx: RunContext[ResearcherDeps],
    source_id: str,
) -> dict:
    """DEEP-RESEARCH only: commit the anchor story BEFORE searching for supporting sources.

    This is a required gate in deep mode. Once `check_relevance` has confirmed a
    relevant, non-duplicate candidate, call `choose_anchor` with that candidate's
    id to lock it in as the anchor — the single story this post is about. You MUST
    do this before you start hunting for supporting sources; `gather_supporting_sources`
    refuses to run until an anchor is committed.

    The `source_id` MUST be a candidate you fetched this tick that came back
    `is_relevant` from `check_relevance` (fresh AND not a duplicate). Pick the
    highest-significance relevant candidate. You can only commit one anchor; calling
    this again replaces the previous choice.

    Args:
        source_id: the id (e.g. 's1') of the relevant candidate to anchor on.
    """
    if ctx.deps.channel.research_depth != "deep":
        return {
            "error": (
                "choose_anchor is only available on deep-research channels. This "
                "channel is single-source: return the chosen source as picked_id."
            )
        }
    cand = ctx.deps.fetched.get(source_id)
    if cand is None:
        return {
            "error": f"{source_id} was never fetched this tick.",
            "fetched_ids": list(ctx.deps.fetched.keys()),
        }
    if not cand.was_checked:
        return {
            "error": (
                f"{source_id} has not been through check_relevance yet. Run "
                "check_relevance first, then anchor on a relevant candidate."
            )
        }
    if not cand.is_relevant:
        return {
            "error": (
                f"{source_id} did not pass check_relevance (stale or duplicate) and "
                "cannot be the anchor. Choose a candidate that came back is_relevant."
            )
        }
    ctx.deps.anchor_id = source_id
    return {
        "anchor_id": source_id,
        "title": cand.title or "",
        "message": (
            "Anchor committed. Now search for supporting sources covering THIS same "
            "story (other outlets, the primary announcement, supporting data), fetch "
            "them, then call gather_supporting_sources with their ids."
        ),
    }


async def gather_supporting_sources(
    ctx: RunContext[ResearcherDeps],
    source_ids: list[str],
    freshness_days: Optional[int] = None,
) -> dict:
    """DEEP-RESEARCH only: commit the supporting sources that back the anchor.

    Call this AFTER `choose_anchor` and after you have fetched the extra sources you
    want to use. Pass `source_ids` — the list of fetched-source ids you decided to
    attach to the anchor. Each is freshness-checked (publish date within the window);
    fresh ids are accepted and handed to the writer. It does NOT run duplicate/
    collision detection — supporting sources corroborate the anchor, so overlap
    between them is fine. Only the anchor must be non-duplicate.

    The anchor id is rejected if you include it, ids you never fetched are rejected,
    and the accepted set is capped at `deep_max_sources`. Returns the accepted and
    rejected ids (with reasons). Available only on deep-research channels.

    Args:
        source_ids: ids (e.g. ['s2', 's4']) of the fetched supporting sources to use.
        freshness_days: optional tighter window; falls back to the channel default.
    """
    if ctx.deps.channel.research_depth != "deep":
        return {
            "error": (
                "gather_supporting_sources is only available on deep-research "
                "channels. This channel is single-source: return the chosen source "
                "as picked_id."
            )
        }
    if ctx.deps.anchor_id is None:
        return {
            "error": (
                "No anchor committed yet. Call choose_anchor with your relevant "
                "anchor candidate before gathering supporting sources."
            )
        }

    window_days = (
        freshness_days
        if freshness_days is not None
        else (ctx.deps.channel.search_freshness_days or DEFAULT_RELEVANCE_DAYS)
    )
    now = datetime.now(timezone.utc)

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for sid in source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        if sid == ctx.deps.anchor_id:
            rejected.append({"id": sid, "reason": "is_anchor"})
            continue
        cand = ctx.deps.fetched.get(sid)
        if cand is None:
            rejected.append({"id": sid, "reason": "not_fetched"})
            continue
        date_info = _date_verdict(cand.published_at, window_days, now)
        cand.fresh_checked = True
        cand.is_fresh = bool(date_info["within_window"])
        if not cand.is_fresh:
            rejected.append({"id": sid, "reason": date_info["date_verdict"]})
            continue
        if len(accepted) >= ctx.deps.deep_max_sources:
            rejected.append({"id": sid, "reason": "over_deep_max_sources"})
            continue
        accepted.append({"id": sid, "url": cand.url, "title": cand.title or ""})

    ctx.deps.supporting_ids = [a["id"] for a in accepted]

    if accepted:
        label = ", ".join(f"{a['id']} — {a['title']}" for a in accepted)
        message = (
            f"{len(accepted)} supporting source(s) committed: {label}. "
            "Finish by returning the anchor as picked_id."
        )
    else:
        message = (
            "No supporting sources committed (none were fresh/fetched). The post "
            "will be written from the anchor alone — return it as picked_id."
        )

    return {
        "freshness_days": window_days,
        "anchor_id": ctx.deps.anchor_id,
        "accepted": accepted,
        "rejected": rejected,
        "deep_max_sources": ctx.deps.deep_max_sources,
        "message": message,
    }


def build_researcher_agent(
    *, native_search: bool
) -> Agent[ResearcherDeps, ResearcherOutput]:
    """Build a Researcher agent for the chosen provider's web-search capability.

    `native_search=True` attaches OpenAI's built-in `WebSearchTool` (Responses
    API). `native_search=False` is for providers without native web search
    (Fireworks/GLM and other `OpenAIChatModel` endpoints): the built-in is
    omitted and a regular `web_search` function tool (DuckDuckGo-backed) is
    registered instead. The `fetch_url` / `check_relevance` loop is identical in
    both cases.
    """
    agent: Agent[ResearcherDeps, ResearcherOutput] = Agent(
        # Explicit name so the instrumentation/agent-run span is labelled
        # "researcher" regardless of how the agent is invoked. Without it,
        # PydanticAI infers the name from the caller's variable (the orchestrator
        # holds it in a local `agent`), which mislabels the span.
        name="researcher",
        output_type=ResearcherOutput,
        deps_type=ResearcherDeps,
        builtin_tools=[WebSearchTool()] if native_search else [],
        instrument=True,
        retries=1,
    )
    agent.system_prompt(_system_prompt)
    if not native_search:
        agent.tool(web_search)
    agent.tool(fetch_url)
    agent.tool(check_relevance)
    # Deep-research channels commit an anchor (choose_anchor) and then attach
    # supporting sources (gather_supporting_sources). Both no-op (return an error)
    # on single-source channels, so they are safe to register on both variants.
    agent.tool(choose_anchor)
    agent.tool(gather_supporting_sources)
    return agent


# Native (OpenAI Responses) researcher — default. The fallback variant is used
# for providers without native web search.
researcher_agent = build_researcher_agent(native_search=True)
researcher_agent_fallback = build_researcher_agent(native_search=False)
