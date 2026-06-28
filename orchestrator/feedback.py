from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from agents.deps import PromptRefinerDeps
from agents.history import build_message_history
from agents.prompt_refiner import (
    PromptRefinerOutput,
    prompt_refiner_agent,
    render_system_prompt,
)
from llm.models import ModelFactory
from orchestrator.models import AppConfig
from orchestrator.orchestrator import _html_escape, md_to_html
from storage.repositories import (
    ChannelRepo,
    MessageRepo,
    PendingPromptRepo,
    RefinementSessionRepo,
)
from telegram_client.client import InlineButton, TelegramBotClient

log = logging.getLogger(__name__)


@dataclass
class FeedbackState:
    """In-memory pending state per (chat) — what the bot is currently waiting for."""
    awaiting_feedback_for: Optional[tuple[str, int, str]] = None  # (channel_id, db_message_id, session_id)


class FeedbackFlow:
    def __init__(
        self,
        cfg: AppConfig,
        tg: TelegramBotClient,
        models: ModelFactory,
        channels: ChannelRepo,
        messages: MessageRepo,
        pending: PendingPromptRepo,
        sessions: RefinementSessionRepo,
        on_channel_saved: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        self.cfg = cfg
        self.tg = tg
        self.models = models
        self.channels = channels
        self.messages = messages
        self.pending = pending
        self.sessions = sessions
        self.on_channel_saved = on_channel_saved
        self.state = FeedbackState()

    # ----- entry points -----

    async def on_feedback_button(self, channel_id: str, db_message_id: int) -> None:
        session_id = f"{channel_id}:{db_message_id}"
        await self.sessions.open(session_id, channel_id, db_message_id)
        self.state.awaiting_feedback_for = (channel_id, db_message_id, session_id)
        await self.tg.send_message(
            f"What should I change for <b>{_html_escape(channel_id)}</b>? Reply to this message.",
            force_reply=True,
        )

    async def on_user_text(self, text: str) -> bool:
        """Returns True if the text was consumed by the feedback flow."""
        if self.state.awaiting_feedback_for is None:
            return False
        channel_id, db_message_id, session_id = self.state.awaiting_feedback_for
        self.state.awaiting_feedback_for = None
        await self._refine_once(channel_id, db_message_id, session_id, text)
        return True

    async def on_approve(self, pending_id: int) -> None:
        row = await self._load_pending(pending_id)
        if row is None or row["status"] != "pending":
            await self.tg.send_message("That proposal is no longer pending.")
            return
        channel_id = row["channel_id"]
        await self.channels.set_topic_prompt(channel_id, row["proposed_prompt"])
        extra_lines: list[str] = []

        proposed_freshness = row["proposed_freshness_days"]
        if proposed_freshness is not None:
            await self.channels.set_freshness_days(channel_id, proposed_freshness)
            extra_lines.append(
                f"<i>Freshness:</i> now only items published in the last "
                f"{proposed_freshness} day(s)"
            )

        schedule_changed = row["proposed_schedule_kind"] is not None
        if schedule_changed:
            spec = json.loads(row["proposed_schedule_spec"] or "{}")
            await self.channels.set_schedule(
                channel_id, row["proposed_schedule_kind"], spec
            )
            extra_lines.append(
                f"<i>Schedule:</i> now <code>{row['proposed_schedule_kind']} {spec}</code>"
            )

        if row["proposed_format"] is not None:
            new_format = row["proposed_format"] or None
            await self.channels.set_format(channel_id, new_format)
            extra_lines.append(
                "<i>Format:</i> reset to default"
                if new_format is None
                else "<i>Format:</i> updated"
            )

        if row["proposed_research_prompt"] is not None:
            await self.channels.set_research_prompt(
                channel_id, row["proposed_research_prompt"]
            )
            extra_lines.append("<i>Research brief:</i> updated")

        await self.pending.resolve(pending_id, "approved")
        await self.sessions.close(row["session_id"], "approved")
        if schedule_changed and self.on_channel_saved is not None:
            await self.on_channel_saved(channel_id)

        tail = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
        await self.tg.send_message(
            f"Saved changes for <b>{_html_escape(channel_id)}</b> ✅\n<i>{md_to_html(row['change_summary'])}</i>{tail}"
        )

    async def on_cancel(self, pending_id: int) -> None:
        row = await self._load_pending(pending_id)
        if row is None:
            return
        await self.pending.resolve(pending_id, "cancelled")
        await self.sessions.close(row["session_id"], "cancelled")
        await self.tg.send_message("Cancelled — current prompt kept.")

    async def on_more_feedback(self, pending_id: int) -> None:
        row = await self._load_pending(pending_id)
        if row is None:
            return
        self.state.awaiting_feedback_for = (
            row["channel_id"],
            row["triggered_by_message_id"] or 0,
            row["session_id"],
        )
        await self.tg.send_message("Tell me what to change about this proposal. Reply here.", force_reply=True)

    # ----- core -----

    async def _refine_once(
        self,
        channel_id: str,
        db_message_id: int,
        session_id: str,
        user_feedback: str,
    ) -> None:
        channel = await self.channels.get(channel_id)
        if channel is None:
            await self.tg.send_message(f"Channel <code>{_html_escape(channel_id)}</code> not found.")
            return

        triggered_message = None
        if db_message_id:
            msg = await self.messages.get(db_message_id)
            if msg:
                triggered_message = {
                    "title": msg.title,
                    "body": msg.body,
                    "source_urls": msg.source_urls,
                }

        history = await self.pending.session_history(session_id)
        recent_accepted = await self.pending.recent_accepted_for_channel(channel.id, n=5)

        deps = PromptRefinerDeps(
            channel=channel,
            current_prompt=channel.topic_prompt_active,
            current_research_prompt=channel.research_prompt,
            triggered_message=triggered_message,
            user_feedback=user_feedback,
            session_history=[
                {
                    "user_feedback_text": h["user_feedback_text"],
                    "proposed_prompt": h["proposed_prompt"],
                }
                for h in history
            ],
            recent_accepted=[
                {"change_summary": r["change_summary"], "resolved_at": r["resolved_at"]}
                for r in recent_accepted
            ],
        )

        prior_turns: list[dict] = []
        for h in deps.session_history:
            prior_turns.append({"role": "user", "text": h["user_feedback_text"]})
            prior_turns.append({"role": "assistant", "text": h["proposed_prompt"]})
        message_history = build_message_history(
            render_system_prompt(deps), prior_turns
        )

        model = self.models.get(self.cfg.model_for("refiner"))
        try:
            result = await prompt_refiner_agent.run(
                user_prompt=user_feedback,
                deps=deps,
                model=model,
                message_history=message_history,
            )
        except Exception as e:
            log.exception("Prompt refiner failed: %s", e)
            await self.tg.send_message(f"Refinement failed: {e}")
            return

        out: PromptRefinerOutput = result.output
        proposed_prompt = deps.working_prompt
        proposed_freshness = (
            deps.proposed_freshness_days if deps.freshness_changed else None
        )
        proposed_schedule_kind = (
            deps.proposed_schedule_kind if deps.schedule_changed else None
        )
        proposed_schedule_spec = (
            deps.proposed_schedule_spec if deps.schedule_changed else None
        )
        proposed_format = deps.proposed_format if deps.format_changed else None
        proposed_research_prompt = (
            deps.working_research_prompt if deps.research_prompt_changed else None
        )
        pending_id = await self.pending.insert(
            session_id=session_id,
            channel_id=channel.id,
            triggered_by_message_id=db_message_id or None,
            user_feedback_text=user_feedback,
            proposed_prompt=proposed_prompt,
            change_summary=out.change_summary,
            proposed_freshness_days=proposed_freshness,
            proposed_schedule_kind=proposed_schedule_kind,
            proposed_schedule_spec=proposed_schedule_spec,
            proposed_format=proposed_format,
            proposed_research_prompt=proposed_research_prompt,
        )

        change_lines = ""
        if proposed_freshness is not None:
            change_lines += (
                f"<i>Freshness:</i> only items published in the last "
                f"{proposed_freshness} day(s)\n"
            )
        if deps.schedule_changed:
            change_lines += (
                f"<i>Schedule:</i> <code>{proposed_schedule_kind} {proposed_schedule_spec}</code>\n"
            )
        if deps.format_changed:
            change_lines += (
                "<i>Format:</i> reset to default\n"
                if not proposed_format
                else f"<i>Format:</i>\n<pre>{_html_escape(proposed_format)}</pre>\n"
            )
        if proposed_research_prompt is not None:
            change_lines += (
                f"<i>Research brief:</i>\n<pre>{_html_escape(proposed_research_prompt)}</pre>\n"
            )

        await self.tg.send_message(
            f"<b>Proposed changes for {_html_escape(channel.display_name)}</b>\n"
            f"<i>Change:</i> {md_to_html(out.change_summary)}\n"
            f"{change_lines}\n"
            f"<b>Prompt:</b>\n<pre>{_html_escape(proposed_prompt)}</pre>",
            buttons=[
                InlineButton("✅ Approve", callback_data=f"prompt:approve:{pending_id}"),
                InlineButton("✏️ More feedback", callback_data=f"prompt:more:{pending_id}"),
                InlineButton("❌ Cancel", callback_data=f"prompt:cancel:{pending_id}"),
            ],
        )

    async def _load_pending(self, pending_id: int) -> Optional[dict]:
        row = await self.pending.db.fetchone(
            "SELECT * FROM pending_prompt_updates WHERE id=?", (pending_id,)
        )
        return dict(row) if row else None
