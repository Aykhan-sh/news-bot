from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.deps import NewsQADeps
from agents.history import build_message_history
from agents.news_qa import news_qa_agent, render_system_prompt
from llm.models import ModelFactory
from orchestrator.models import AppConfig
from orchestrator.orchestrator import md_to_html
from storage.repositories import ChannelRepo, MessageRepo
from telegram_client.client import InlineButton, TelegramBotClient

log = logging.getLogger(__name__)


@dataclass
class QAState:
    active: bool = False
    channel_id: Optional[str] = None
    db_message_id: Optional[int] = None
    awaiting_question: bool = False
    history: list[dict] = field(default_factory=list)


class NewsQAFlow:
    """Lets the user ask follow-up questions about a posted news item."""

    def __init__(
        self,
        cfg: AppConfig,
        tg: TelegramBotClient,
        models: ModelFactory,
        channels: ChannelRepo,
        messages: MessageRepo,
    ) -> None:
        self.cfg = cfg
        self.tg = tg
        self.models = models
        self.channels = channels
        self.messages = messages
        self.state = QAState()

    def is_active(self) -> bool:
        return self.state.active and self.state.awaiting_question

    async def on_ask_button(self, channel_id: str, db_message_id: int) -> None:
        self.state = QAState(
            active=True,
            channel_id=channel_id,
            db_message_id=db_message_id,
            awaiting_question=True,
        )
        await self.tg.send_message(
            f"Ask anything about that post from <b>{channel_id}</b>. Reply here with your question.",
            force_reply=True,
        )

    async def on_user_text(self, text: str) -> bool:
        if not self.state.active or not self.state.awaiting_question:
            return False
        assert self.state.channel_id is not None and self.state.db_message_id is not None
        channel_id = self.state.channel_id
        db_message_id = self.state.db_message_id
        self.state.awaiting_question = False

        ch = await self.channels.get(channel_id)
        msg = await self.messages.get(db_message_id)
        if ch is None or msg is None:
            await self.tg.send_message("Couldn't find that post anymore.")
            self.state = QAState()
            return True

        post = {
            "title": msg.title,
            "body": msg.body,
            "hashtags": msg.hashtags,
            "source_urls": msg.source_urls,
        }

        deps = NewsQADeps(channel=ch, post=post)
        message_history = build_message_history(
            render_system_prompt(deps), list(self.state.history)
        )
        self.state.history.append({"role": "user", "text": text})

        model = self.models.get(self.cfg.model_for("writer"))
        try:
            result = await news_qa_agent.run(
                user_prompt=text,
                deps=deps,
                model=model,
                message_history=message_history,
            )
        except Exception as e:
            log.exception("News QA failed: %s", e)
            await self.tg.send_message(f"Sorry, couldn't answer: <code>{e}</code>")
            self.state.awaiting_question = True
            return True

        answer = result.output.answer
        self.state.history.append({"role": "assistant", "text": answer})

        await self.tg.send_message(
            md_to_html(answer),
            buttons=[
                InlineButton("💬 Ask again", f"ask:{channel_id}:{db_message_id}"),
                InlineButton("✅ Done", "ask:done"),
            ],
        )
        return True

    async def on_done(self) -> None:
        self.state = QAState()
        await self.tg.send_message("<i>Closed the Q&amp;A.</i>")
