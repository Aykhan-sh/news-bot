from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.deps import WriterDeps
from orchestrator.prompt_builder import render
from sources.fetch import fetch_url_excerpt

log = logging.getLogger(__name__)


class WriterOutput(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    body: str = Field(min_length=20, max_length=3500)
    keywords: list[str] = Field(min_length=2, max_length=8)
    hashtags: list[str] = Field(default_factory=list, max_length=5)
    sources_used: list[str] = Field(default_factory=list, max_length=8)
    suggest_image: bool = False
    image_brief: Optional[str] = None


writer_agent: Agent[WriterDeps, WriterOutput] = Agent(
    name="writer",
    output_type=WriterOutput,
    deps_type=WriterDeps,
    instrument=True,
    retries=1,
)


@writer_agent.system_prompt
async def _system_prompt(ctx: RunContext[WriterDeps]) -> str:
    return render(
        "writer.system.j2",
        channel=ctx.deps.channel,
        window=[
            {"title": m.title, "keywords": m.keywords}
            for m in ctx.deps.window
        ],
        research_note=ctx.deps.research_note,
    )


@writer_agent.tool
async def fetch_url(
    ctx: RunContext[WriterDeps],
    url: str,
    max_chars: int = 12000,
) -> dict:
    """Re-open the chosen source for more text than the research note carries.

    The research note already includes the excerpt the Researcher pulled, but it
    was capped at a smaller limit. Call this with the source `url` only when that
    excerpt is too thin to write an accurate post — it returns a wider, cleaned
    excerpt. Capped per tick; do not fetch unrelated pages.
    """
    if ctx.deps.fetch_calls_made >= ctx.deps.fetch_budget:
        return {"error": f"fetch budget exhausted ({ctx.deps.fetch_budget})"}
    ctx.deps.fetch_calls_made += 1
    excerpt = await fetch_url_excerpt(url, max_chars=max_chars)
    return excerpt.to_dict()
