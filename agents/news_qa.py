from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.deps import NewsQADeps
from orchestrator.prompt_builder import render

log = logging.getLogger(__name__)


class NewsQAOutput(BaseModel):
    answer: str = Field(min_length=1, max_length=3500)


news_qa_agent: Agent[NewsQADeps, NewsQAOutput] = Agent(
    output_type=NewsQAOutput,
    deps_type=NewsQADeps,
    instrument=True,
    retries=1,
)


def render_system_prompt(deps: NewsQADeps) -> str:
    return render(
        "news_qa.j2",
        channel=deps.channel,
        post=deps.post,
    )
