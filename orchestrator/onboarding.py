from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from agents.deps import SetupAgentDeps
from agents.history import build_message_history
from agents.setup_agent import (
    ProposedChannel,
    SetupAgentOutput,
    render_system_prompt,
    setup_agent,
)
from llm.models import ModelFactory
from orchestrator.models import AppConfig
from orchestrator.orchestrator import _html_escape, md_to_html
from storage.repositories import ChannelRepo
from telegram_client.client import InlineButton, TelegramBotClient

log = logging.getLogger(__name__)


@dataclass
class OnboardingState:
    active: bool = False
    history: list[dict] = field(default_factory=list)
    awaiting_reply: bool = False
    last_proposal: Optional[ProposedChannel] = None
    user_locale_hint: Optional[str] = None
    editing_channel_id: Optional[str] = None


class OnboardingFlow:
    """Conversational `/addchannel` flow.

    The user describes — in plain language — the channel they want.
    The setup-assistant LLM proposes a complete channel spec, asks clarifying
    questions when needed, and the user approves / refines / cancels.
    """

    def __init__(
        self,
        cfg: AppConfig,
        tg: TelegramBotClient,
        models: ModelFactory,
        channels: ChannelRepo,
        channels_dir: Path,
        on_channel_saved: Callable[[str], Awaitable[None]],
    ) -> None:
        self.cfg = cfg
        self.tg = tg
        self.models = models
        self.channels = channels
        self.channels_dir = channels_dir
        self.on_channel_saved = on_channel_saved
        self.state = OnboardingState()

    # ----- entry points -----

    async def start(self, user_locale_hint: Optional[str] = None) -> None:
        self.state = OnboardingState(
            active=True, awaiting_reply=True, user_locale_hint=user_locale_hint
        )
        await self.tg.send_message(
            "Let's set up a new channel. <b>Describe what you want</b> in your own words.\n\n"
            "Include things like:\n"
            "• <b>Topic</b> — what should it be about?\n"
            "• <b>Style / format</b> — bullets? a single word? a short news brief?\n"
            "• <b>How often</b> — daily, every few hours, a few times a day…?\n"
            "• <b>Sources</b> — should it pull from the live web, or just use the model's own knowledge?\n\n"
            'Example: <i>"Send me one new Kazakh word per day with the meaning and an example sentence."</i>',
            force_reply=True,
        )

    async def start_edit(
        self, channel_id: str, user_locale_hint: Optional[str] = None
    ) -> None:
        ch = await self.channels.get(channel_id)
        if ch is None:
            await self.tg.send_message(f"Channel <code>{_html_escape(channel_id)}</code> not found.")
            return
        self.state = OnboardingState(
            active=True,
            awaiting_reply=True,
            user_locale_hint=user_locale_hint,
            editing_channel_id=channel_id,
        )
        sched = f"{ch.schedule_kind} {ch.schedule_spec}"
        if ch.format and ch.format.strip():
            fmt_line = f"• <b>current format:</b>\n<pre>{_html_escape(ch.format.strip())}</pre>\n"
        else:
            fmt_line = "• <b>current format:</b> <code>default</code>\n"
        await self.tg.send_message(
            f"Editing <code>{_html_escape(ch.id)}</code> — {_html_escape(ch.display_name)}.\n\n"
            f"• <b>current schedule:</b> <code>{_html_escape(sched)}</code>\n"
            f"{fmt_line}"
            f"• <b>current prompt:</b>\n<pre>{_html_escape(ch.topic_prompt_active)}</pre>\n\n"
            "Tell me what to change in your own words — schedule, prompt/topic, format, "
            "or anything else. Examples:\n"
            '• <i>"Fire every day at 9am Moscow time instead."</i>\n'
            '• <i>"Make it focus on AI safety news only, skip product launches."</i>\n'
            '• <i>"Both — change schedule to 8am GMT+5 and shorten the prompt."</i>',
            force_reply=True,
        )

    def is_active(self) -> bool:
        return self.state.active

    async def cancel(self) -> None:
        self.state = OnboardingState()
        await self.tg.send_message("Onboarding cancelled.")

    async def on_user_text(
        self, text: str, user_locale_hint: Optional[str] = None
    ) -> bool:
        """Returns True if the text was consumed by the onboarding flow."""
        if not self.state.active or not self.state.awaiting_reply:
            return False
        if user_locale_hint and not self.state.user_locale_hint:
            self.state.user_locale_hint = user_locale_hint
        self.state.awaiting_reply = False
        await self._run_assistant(text)
        return True

    async def on_approve(self) -> None:
        if not self.state.active or self.state.last_proposal is None:
            await self.tg.send_message("No pending channel proposal.")
            return
        proposal = self.state.last_proposal
        is_edit = self.state.editing_channel_id is not None
        if is_edit:
            proposal.id = self.state.editing_channel_id  # enforce stable id
        spec = self._proposal_to_spec(proposal)
        try:
            await self.channels.upsert_from_yaml(spec)
            if is_edit:
                await self.channels.set_topic_prompt(spec["id"], spec["topic_prompt"])
                await self.channels.set_research_prompt(
                    spec["id"], spec.get("research_prompt")
                )
        except Exception as e:
            log.exception("Failed to save channel: %s", e)
            await self.tg.send_message(f"Failed to save channel: <code>{_html_escape(str(e))}</code>")
            return
        yaml_path = self.channels_dir / f"{spec['id']}.yaml"
        try:
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to write %s: %s", yaml_path, e)
        await self.on_channel_saved(spec["id"])
        self.state = OnboardingState()
        verb = "Updated" if is_edit else "Created"
        await self.tg.send_message(
            f"{verb} channel <code>{_html_escape(spec['id'])}</code> ✅\n"
            f"Use /channels to manage it, or /now to fire it immediately."
        )

    async def on_more_feedback(self) -> None:
        if not self.state.active:
            return
        self.state.awaiting_reply = True
        await self.tg.send_message(
            "Tell me what to change. Reply here.",
            force_reply=True,
        )

    # ----- internals -----

    async def _run_assistant(self, user_text: str) -> None:
        self.state.history.append({"role": "user", "text": user_text})

        existing = await self.channels.list_all()

        try:
            server_tz = time.tzname[0] if time.tzname else None
            offset_h = -time.timezone / 3600
            sign = "-" if offset_h >= 0 else "+"
            server_tz = f"{server_tz} (Etc/GMT{sign}{int(abs(offset_h))})"
        except Exception:
            server_tz = None

        editing_channel = None
        if self.state.editing_channel_id:
            ch = await self.channels.get(self.state.editing_channel_id)
            if ch is not None:
                editing_channel = {
                    "id": ch.id,
                    "display_name": ch.display_name,
                    "hashtag": ch.hashtag,
                    "mode": ch.mode,
                    "format": ch.format,
                    "schedule_kind": ch.schedule_kind,
                    "schedule_spec": ch.schedule_spec,
                    "search_freshness_days": ch.search_freshness_days,
                    "topic_prompt_active": ch.topic_prompt_active,
                    "research_prompt": ch.research_prompt,
                }

        deps = SetupAgentDeps(
            existing_channels=existing,
            user_locale_hint=self.state.user_locale_hint,
            server_timezone=server_tz,
            editing_channel=editing_channel,
        )

        message_history = build_message_history(
            render_system_prompt(deps), self.state.history[:-1]
        )

        model = self.models.get(self.cfg.model_for("setup"))
        try:
            result = await setup_agent.run(
                user_prompt=user_text,
                deps=deps,
                model=model,
                message_history=message_history,
            )
        except Exception as e:
            log.exception("Setup assistant failed: %s", e)
            await self.tg.send_message(f"Setup failed: <code>{_html_escape(str(e))}</code>")
            self.state.awaiting_reply = True
            return

        out: SetupAgentOutput = result.output
        self.state.history.append({"role": "assistant", "text": out.assistant_message})

        if out.ready_to_save and out.proposed_channel is not None:
            self.state.last_proposal = out.proposed_channel
            await self._send_proposal(out)
            return

        # Still gathering info — show message + clarifying questions, await next reply.
        body = md_to_html(out.assistant_message.strip())
        if out.clarifying_questions:
            body += "\n\n" + "\n".join(f"• {md_to_html(q)}" for q in out.clarifying_questions)
        self.state.awaiting_reply = True
        await self.tg.send_message(body, force_reply=True)

    async def _send_proposal(self, out: SetupAgentOutput) -> None:
        assert out.proposed_channel is not None
        p = out.proposed_channel
        sched = f"{p.schedule.kind} {p.schedule.spec}"
        if p.format and p.format.strip():
            fmt_line = f"• <b>format:</b>\n<pre>{_html_escape(p.format.strip())}</pre>\n"
        else:
            fmt_line = "• <b>format:</b> <code>default</code>\n"
        freshness_line = ""
        if p.mode == "sourced":
            if p.freshness_days:
                freshness_line = (
                    f"• <b>freshness:</b> only items published in the last "
                    f"{p.freshness_days} day(s)\n"
                )
            else:
                freshness_line = "• <b>freshness:</b> last 7 day(s) (default)\n"
        summary = (
            f"<b>Proposed channel</b>\n"
            f"<i>{md_to_html(out.assistant_message)}</i>\n\n"
            f"• <b>id:</b> <code>{_html_escape(p.id)}</code>\n"
            f"• <b>name:</b> {_html_escape(p.display_name)}\n"
            f"• <b>hashtag:</b> <code>{_html_escape(p.hashtag)}</code>\n"
            f"• <b>mode:</b> {_html_escape(p.mode)}\n"
            f"• <b>schedule:</b> <code>{_html_escape(sched)}</code>\n"
            f"{freshness_line}"
            f"{fmt_line}\n"
            f"<b>Prompt:</b>\n<pre>{_html_escape(p.topic_prompt)}</pre>"
        )
        if p.mode == "sourced" and p.research_prompt and p.research_prompt.strip():
            summary += f"\n<b>Research brief:</b>\n<pre>{_html_escape(p.research_prompt.strip())}</pre>"
        await self.tg.send_message(
            summary,
            buttons=[
                InlineButton("✅ Create channel", callback_data="setup:approve"),
                InlineButton("✏️ Change something", callback_data="setup:more"),
                InlineButton("❌ Cancel", callback_data="setup:cancel"),
            ],
        )

    def _proposal_to_spec(self, p: ProposedChannel) -> dict:
        spec: dict = {
            "id": p.id,
            "display_name": p.display_name,
            "hashtag": p.hashtag,
            "mode": p.mode,
            "topic_prompt": p.topic_prompt,
            "schedule": {"kind": p.schedule.kind, "spec": p.schedule.spec},
            "model_writer": self.cfg.model_for("writer"),
        }
        if p.format and p.format.strip():
            spec["format"] = p.format.strip()
        if p.mode == "sourced":
            spec["model_researcher"] = self.cfg.model_for("researcher")
            spec["search"] = {"freshness_days": p.freshness_days}
            if p.research_prompt and p.research_prompt.strip():
                spec["research_prompt"] = p.research_prompt.strip()
        return spec
