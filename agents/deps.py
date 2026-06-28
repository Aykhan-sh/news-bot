from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from orchestrator.dedup import DedupEngine
from sources.search import DuckDuckGoSearch, WebSearch
from storage.repositories import ChannelRow, MessageRepo


@dataclass
class Candidate:
    id: str
    url: str
    title: Optional[str] = None
    body: str = ""
    published_at: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    byline: Optional[str] = None
    was_checked: bool = False
    is_relevant: bool = False
    # Deep mode: set by `gather_supporting_sources` (freshness-only gate). A fresh
    # candidate is eligible as a supporting source even if it duplicates another
    # supporting source — only the anchor must pass the full `check_relevance` gate.
    fresh_checked: bool = False
    is_fresh: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "text": self.body,
            "published_at": self.published_at,
            "keywords": self.keywords,
            "byline": self.byline,
        }


@dataclass
class ResearcherDeps:
    channel: ChannelRow
    messages: MessageRepo
    dedup: DedupEngine
    window: list  # list[MessageRow]
    fetch_budget: int
    # Hard cap on `check_relevance` calls this tick (collision-check rounds).
    check_budget: int = 3
    # Deep mode only: most supporting sources (beyond the anchor) the researcher may
    # attach to the writer's note. Mirrors `ResearcherConfig.deep_max_sources`.
    deep_max_sources: int = 4
    # per-run mutable state (tracked by tools)
    fetch_calls_made: int = 0
    check_calls_made: int = 0
    # Registry of candidates fetched this tick, keyed by the id `fetch_url` hands
    # out (e.g. "s1"). The researcher returns one of these ids as `picked_id`, and
    # the orchestrator hands the matching record to the writer.
    fetched: dict[str, Candidate] = field(default_factory=dict)
    # Embeddings computed by `check_relevance` for each fetched candidate, keyed by
    # source id (e.g. "s1"). The orchestrator reuses the picked candidate's vector
    # when saving the post embedding instead of re-embedding the writer's output.
    fetched_vectors: dict[str, list[float]] = field(default_factory=dict)
    # Deep mode: the anchor source the researcher committed to via the `choose_anchor`
    # tool, before it searched for supporting sources. The orchestrator treats this as
    # authoritative (falling back to the final output's `picked_id` only if unset).
    anchor_id: Optional[str] = None
    # Deep mode: supporting source ids accepted by `gather_supporting_sources` (fetched
    # this tick and fresh). The orchestrator hands these to the writer alongside the
    # anchor. Falls back to the final output's `supporting_ids` only if unset.
    supporting_ids: list[str] = field(default_factory=list)
    # Web-search backend used by the fallback `web_search` tool when the researcher
    # runs on a provider without native web search (e.g. Fireworks). Ignored when
    # the researcher uses OpenAI's native `WebSearchTool`.
    web_search: WebSearch = field(default_factory=DuckDuckGoSearch)


@dataclass
class WriterDeps:
    channel: ChannelRow
    window: list  # list[MessageRow]
    research_note: Optional[dict]  # fetched source record {id,url,title,published_at,text}
    # Deep mode: extra fetched source records ({id,url,title,published_at,text}) that
    # back the anchor `research_note`. The writer synthesises all of them into one
    # post and cites the URLs it relied on. Empty for single-source channels.
    supporting_notes: list[dict] = field(default_factory=list)
    # Budget for the writer's own fetch_url tool, used to re-open the chosen
    # source with a wider char limit when the researcher's excerpt is too thin.
    fetch_budget: int = 0
    fetch_calls_made: int = 0


@dataclass
class PromptRefinerDeps:
    channel: ChannelRow
    current_prompt: str
    triggered_message: Optional[dict]
    user_feedback: str
    session_history: list[dict]
    recent_accepted: list[dict]
    # Working copy the `edit_prompt` tool mutates in place. Initialised to
    # `current_prompt`; after the run it holds the proposed prompt.
    working_prompt: str = ""
    # Deterministic content-recency window the `set_freshness_days` tool mutates.
    # Initialised to the channel's current value; `freshness_changed` flips to True
    # only if the tool actually changes it.
    proposed_freshness_days: Optional[int] = None
    freshness_changed: bool = False
    # Schedule (timing + timezone) the `set_schedule` tool mutates. Initialised to
    # the channel's current schedule; `schedule_changed` flips to True only if the
    # tool actually changes it.
    proposed_schedule_kind: Optional[str] = None
    proposed_schedule_spec: Optional[dict] = None
    schedule_changed: bool = False
    # Free-text format instructions the `set_format` tool mutates. Initialised to
    # the channel's current value; `format_changed` flips to True only if the tool
    # actually changes it (an empty string resets the channel to writer defaults).
    proposed_format: Optional[str] = None
    format_changed: bool = False
    # Researcher-only prompt. `current_research_prompt` is the channel's stored
    # value (None = it falls back to the topic prompt). `working_research_prompt`
    # is the copy the `edit_research_prompt` tool mutates in place;
    # `research_prompt_changed` flips to True only if the tool actually changes it.
    current_research_prompt: Optional[str] = None
    working_research_prompt: str = ""
    research_prompt_changed: bool = False

    def __post_init__(self) -> None:
        if not self.working_prompt:
            self.working_prompt = self.current_prompt
        self.proposed_freshness_days = self.channel.search_freshness_days
        self.proposed_schedule_kind = self.channel.schedule_kind
        self.proposed_schedule_spec = self.channel.schedule_spec
        self.proposed_format = self.channel.format
        if self.current_research_prompt is None:
            self.current_research_prompt = self.channel.research_prompt
        if not self.working_research_prompt:
            self.working_research_prompt = (
                self.current_research_prompt or self.current_prompt
            )


@dataclass
class NewsQADeps:
    channel: ChannelRow
    post: dict  # {title, body, hashtags, source_urls}


@dataclass
class SetupAgentDeps:
    existing_channels: list[ChannelRow]
    user_locale_hint: Optional[str] = None
    server_timezone: Optional[str] = None
    editing_channel: Optional[dict] = None
