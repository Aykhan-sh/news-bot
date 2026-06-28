from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.deps import PromptRefinerDeps
from orchestrator.prompt_builder import render

log = logging.getLogger(__name__)


class PromptRefinerOutput(BaseModel):
    change_summary: str = Field(min_length=2, max_length=300)


prompt_refiner_agent: Agent[PromptRefinerDeps, PromptRefinerOutput] = Agent(
    name="prompt_refiner",
    output_type=PromptRefinerOutput,
    deps_type=PromptRefinerDeps,
    instrument=True,
    retries=1,
)


def render_system_prompt(deps: PromptRefinerDeps) -> str:
    return render(
        "prompt_refiner.j2",
        channel=deps.channel,
        current_prompt=deps.current_prompt,
        current_research_prompt=deps.working_research_prompt,
        triggered_message=deps.triggered_message,
        recent_accepted=deps.recent_accepted,
    )


@prompt_refiner_agent.tool
async def edit_prompt(
    ctx: RunContext[PromptRefinerDeps],
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> dict:
    """Edit the channel prompt in place by replacing a snippet.

    Use this instead of rewriting the whole prompt: pass the exact text to
    replace (`old_string`, copied verbatim from the current prompt) and its
    replacement (`new_string`). Call it once per distinct change. To delete text,
    pass an empty `new_string`. To add a brand-new sentence, replace a nearby
    snippet with that snippet plus the new sentence. The accumulated result
    becomes the proposed prompt.

    Args:
        old_string: exact snippet to find in the current prompt.
        new_string: text to put in its place (empty to delete).
        replace_all: replace every occurrence instead of requiring a unique match.
    """
    prompt = ctx.deps.working_prompt
    occurrences = prompt.count(old_string)
    if not old_string or occurrences == 0:
        return {
            "ok": False,
            "error": "old_string not found in the current prompt; copy it exactly.",
            "current_prompt": prompt,
        }
    if occurrences > 1 and not replace_all:
        return {
            "ok": False,
            "error": (
                f"old_string appears {occurrences} times; add surrounding context "
                "to make it unique, or pass replace_all=true."
            ),
            "current_prompt": prompt,
        }
    updated = (
        prompt.replace(old_string, new_string)
        if replace_all
        else prompt.replace(old_string, new_string, 1)
    )
    ctx.deps.working_prompt = updated
    return {"ok": True, "current_prompt": updated}


@prompt_refiner_agent.tool
async def edit_research_prompt(
    ctx: RunContext[PromptRefinerDeps],
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> dict:
    """Edit the channel's **research prompt** in place by replacing a snippet.

    The research prompt is researcher-only — it controls WHAT the bot hunts for on
    the live web (which sources/outlets to watch, which kinds of stories to
    prioritise or skip), NOT how the post is worded or laid out. Use this tool
    only for `sourced` channels and only when the user's feedback is about
    *sourcing / what stories to surface* ("also watch the Anthropic blog",
    "prioritise primary announcements", "stop surfacing funding-round news").
    For what each post says or its scope, use `edit_prompt`; for post shape/style,
    use `set_format`.

    Works exactly like `edit_prompt`: pass the exact snippet to replace
    (`old_string`, copied verbatim from the current research prompt shown in the
    context) and its replacement (`new_string`, empty to delete). Call once per
    distinct change.

    Args:
        old_string: exact snippet to find in the current research prompt.
        new_string: text to put in its place (empty to delete).
        replace_all: replace every occurrence instead of requiring a unique match.
    """
    prompt = ctx.deps.working_research_prompt
    occurrences = prompt.count(old_string)
    if not old_string or occurrences == 0:
        return {
            "ok": False,
            "error": "old_string not found in the current research prompt; copy it exactly.",
            "current_research_prompt": prompt,
        }
    if occurrences > 1 and not replace_all:
        return {
            "ok": False,
            "error": (
                f"old_string appears {occurrences} times; add surrounding context "
                "to make it unique, or pass replace_all=true."
            ),
            "current_research_prompt": prompt,
        }
    updated = (
        prompt.replace(old_string, new_string)
        if replace_all
        else prompt.replace(old_string, new_string, 1)
    )
    ctx.deps.working_research_prompt = updated
    ctx.deps.research_prompt_changed = True
    return {"ok": True, "current_research_prompt": updated}


@prompt_refiner_agent.tool
async def set_freshness_days(
    ctx: RunContext[PromptRefinerDeps],
    days: int,
) -> dict:
    """Set the channel's content-recency window (max publish-date age, in days).

    Use this for `sourced` channels whenever the user changes how recent items
    must be ("only the last 3 days", "today's news only", "you can go back a
    week"). This is a **deterministic** setting enforced by the bot — do NOT write
    article-age rules into the prompt text; change them here instead.

    Args:
        days: maximum age in days for an item's publish date (>= 1). e.g. 3 means
            only items published within the last 3 days are eligible.
    """
    if ctx.deps.channel.mode != "sourced":
        return {
            "ok": False,
            "error": "freshness_days only applies to sourced channels.",
        }
    if days < 1:
        return {"ok": False, "error": "days must be >= 1."}
    ctx.deps.proposed_freshness_days = days
    ctx.deps.freshness_changed = True
    return {"ok": True, "freshness_days": days}


@prompt_refiner_agent.tool
async def set_schedule(
    ctx: RunContext[PromptRefinerDeps],
    kind: str,
    spec: dict,
) -> dict:
    """Change WHEN the channel posts — its cadence, time-of-day, and timezone.

    Use this whenever the user wants to change posting timing ("post twice a day",
    "move it to 9am", "every 4 hours", "switch to Moscow time"). Do NOT write
    cadence/timing into the prompt text — it is a deterministic setting changed
    here instead.

    Args:
        kind: one of 'interval', 'cron', 'probabilistic'.
        spec: the schedule spec for that kind:
            - 'interval': {'hours': 6} or {'minutes': 30}.
            - 'cron': {'hour': '9,18', 'minute': '0', 'timezone': 'Europe/Moscow'}.
            - 'probabilistic': {'per_day': 3, 'start_hour': 9, 'end_hour': 22,
              'timezone': 'Europe/Moscow'}.
            Whenever the spec has a specific time-of-day (cron hour/minute, or
            probabilistic start_hour/end_hour) you MUST include a valid IANA
            'timezone' (e.g. 'Europe/Moscow', 'Etc/GMT-5'). Never use bare offsets
            like 'GMT+5' — convert them (e.g. 'GMT+5' -> 'Etc/GMT-5').
    """
    if kind not in ("interval", "cron", "probabilistic"):
        return {
            "ok": False,
            "error": "kind must be one of 'interval', 'cron', 'probabilistic'.",
        }
    if not isinstance(spec, dict) or not spec:
        return {"ok": False, "error": "spec must be a non-empty mapping."}
    ctx.deps.proposed_schedule_kind = kind
    ctx.deps.proposed_schedule_spec = spec
    ctx.deps.schedule_changed = True
    return {"ok": True, "kind": kind, "spec": spec}


@prompt_refiner_agent.tool
async def set_format(
    ctx: RunContext[PromptRefinerDeps],
    format: str,
) -> dict:
    """Change HOW each post is structured / styled / laid out (free text).

    Use this whenever the user changes the post's shape, length, or presentation
    ("use three bullet points", "put the word in bold then the meaning", "drop the
    emoji", "make it shorter"). Pass the full updated formatting instructions in
    plain language. Pass an empty string to clear the custom format and fall back
    to the writer's sensible defaults. Do NOT change topic/scope here — use
    `edit_prompt` for that.

    Args:
        format: free-text formatting instructions, or "" to reset to defaults.
    """
    ctx.deps.proposed_format = format.strip()
    ctx.deps.format_changed = True
    return {"ok": True, "format": ctx.deps.proposed_format or "(default)"}
