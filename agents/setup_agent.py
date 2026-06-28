from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.deps import SetupAgentDeps
from orchestrator.prompt_builder import render

log = logging.getLogger(__name__)


class ScheduleSpec(BaseModel):
    kind: str = Field(description="One of: 'interval', 'cron', 'probabilistic'.")
    spec: dict = Field(
        default_factory=dict,
        description=(
            "For 'interval': mapping like {'hours': 6} or {'minutes': 30}. "
            "For 'cron': mapping like {'hour': '9,18', 'minute': '0'}. "
            "For 'probabilistic': mapping like {'per_day': 3}."
        ),
    )


class ProposedChannel(BaseModel):
    id: str = Field(description="Short lowercase slug, e.g. 'ai_news'. ASCII only, [a-z0-9_].")
    display_name: str
    hashtag: str = Field(description="Hashtag including '#', e.g. '#ai_news'.")
    mode: str = Field(description="Either 'sourced' (live web search) or 'llm_only'.")
    topic_prompt: str = Field(description="Free-form description of what the channel should publish.")
    research_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Sourced channels ONLY: a researcher-facing brief describing what to "
            "hunt for on the live web — which sources/outlets to watch, which "
            "kinds of items to prioritise, and what to skip. This is fed ONLY to "
            "the research agent, never the writer, so keep it about finding "
            "stories, not about wording or post layout. Leave null for `llm_only` "
            "channels (no research step) or when the topic_prompt already fully "
            "describes what to look for."
        ),
    )
    format: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text formatting instructions for the post, in the user's "
            "own terms (e.g. 'one bold word, then meaning, then an example sentence', "
            "or 'three short bullet points, no emoji'). Set this ONLY when the user "
            "asked for a specific structure/style/length/layout. Leave null when the "
            "user has no special format request — the writer then uses sensible "
            "defaults."
        ),
    )
    schedule: ScheduleSpec
    freshness_days: Optional[int] = Field(
        default=None,
        description=(
            "Sourced channels ONLY: how fresh an item's publish date must be to count "
            "as news — the max age in days. e.g. 3 = only items published within the "
            "last 3 days. This is a deterministic content-recency gate, separate from "
            "the posting schedule. Always set it for `sourced` channels (default 7 if "
            "the user has no preference). Leave null for `llm_only`."
        ),
    )


class SetupAgentOutput(BaseModel):
    assistant_message: str = Field(
        description=(
            "What to say to the user right now. If asking clarifying questions, "
            "phrase them here in plain conversational language."
        )
    )
    clarifying_questions: list[str] = Field(default_factory=list, max_length=4)
    proposed_channel: Optional[ProposedChannel] = None
    ready_to_save: bool = False


setup_agent: Agent[SetupAgentDeps, SetupAgentOutput] = Agent(
    name="setup_agent",
    output_type=SetupAgentOutput,
    deps_type=SetupAgentDeps,
    instrument=True,
    retries=1,
)


def render_system_prompt(deps: SetupAgentDeps) -> str:
    return render(
        "setup_agent.j2",
        existing_channels=deps.existing_channels,
        user_locale_hint=deps.user_locale_hint,
        server_timezone=deps.server_timezone,
        editing_channel=deps.editing_channel,
    )
