from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from llm.embeddings import EmbeddingService
from llm.models import ModelFactory
from orchestrator.cost import CostController
from orchestrator.dedup import DedupEngine
from orchestrator.feedback import FeedbackFlow
from orchestrator.models import AppConfig, load_channels_dir, load_config
from orchestrator.news_qa import NewsQAFlow
from orchestrator.onboarding import OnboardingFlow
from orchestrator.orchestrator import Orchestrator
from orchestrator.scheduler import ChannelScheduler
from storage.db import init_db
from storage.repositories import (
    ChannelRepo,
    MessageRepo,
    PendingPromptRepo,
    RefinementSessionRepo,
    UsageRepo,
)
from telegram_client.client import InlineButton, TelegramBotClient

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("news-bot")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _maybe_init_laminar(cfg: AppConfig) -> None:
    if not cfg.observability.lmnr_enabled or not cfg.observability.lmnr_project_api_key:
        log.info("Laminar disabled")
        return
    try:
        from lmnr import Laminar

        Laminar.initialize(project_api_key=cfg.observability.lmnr_project_api_key)
        log.info("Laminar initialised")
    except Exception as e:
        log.warning("Laminar init failed: %s", e)


class App:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.tg = TelegramBotClient(cfg.telegram, session_dir=str(ROOT / "data"))
        os.environ.setdefault("OPENAI_API_KEY", cfg.openai.api_key)
        self.models = ModelFactory(cfg.openai)
        self.scheduler: ChannelScheduler | None = None
        self.orchestrator: Orchestrator | None = None
        self.feedback: FeedbackFlow | None = None
        self.news_qa: NewsQAFlow | None = None
        self.onboarding: OnboardingFlow | None = None
        self.channel_repo: ChannelRepo | None = None
        self.message_repo: MessageRepo | None = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        db = await init_db(self.cfg.storage.db_path)

        self.channel_repo = ChannelRepo(db)
        self.message_repo = MessageRepo(db)
        usage_repo = UsageRepo(db)
        pending_repo = PendingPromptRepo(db)
        session_repo = RefinementSessionRepo(db)

        for spec in load_channels_dir(ROOT / "channels"):
            await self.channel_repo.upsert_from_yaml(spec)

        embedder = EmbeddingService(
            self.cfg.openai.api_key, self.cfg.openai.embedding_model
        )
        dedup = DedupEngine(
            self.message_repo,
            embedder,
            threshold=self.cfg.dedup.similarity_threshold,
            lookback_days=self.cfg.dedup.lookback_days,
            log_candidates=self.cfg.dedup.log_candidates,
            candidate_log_path=self.cfg.dedup.candidate_log_path,
        )
        cost = CostController(usage_repo, self.cfg.cost_control)

        self.orchestrator = Orchestrator(
            cfg=self.cfg,
            tg=self.tg,
            models=self.models,
            channels=self.channel_repo,
            messages=self.message_repo,
            usage=usage_repo,
            cost=cost,
            dedup=dedup,
        )

        self.feedback = FeedbackFlow(
            cfg=self.cfg,
            tg=self.tg,
            models=self.models,
            channels=self.channel_repo,
            messages=self.message_repo,
            pending=pending_repo,
            sessions=session_repo,
            usage=usage_repo,
            on_channel_saved=self._on_channel_saved,
        )

        self.news_qa = NewsQAFlow(
            cfg=self.cfg,
            tg=self.tg,
            models=self.models,
            channels=self.channel_repo,
            messages=self.message_repo,
            usage=usage_repo,
        )

        self.onboarding = OnboardingFlow(
            cfg=self.cfg,
            tg=self.tg,
            models=self.models,
            channels=self.channel_repo,
            usage=usage_repo,
            channels_dir=ROOT / "channels",
            on_channel_saved=self._on_channel_saved,
        )

        self.scheduler = ChannelScheduler(
            channels=self.channel_repo,
            fire_callback=self._fire,
            probabilistic_jitter_seconds=self.cfg.scheduling.probabilistic_jitter_seconds,
        )
        await self.scheduler.schedule_all()
        self.scheduler.start()

        await self._update_bot_commands()

        log.info("News bot started. Polling Telegram updates.")
        poll_task = asyncio.create_task(
            self.tg.poll(self._on_message, self._on_callback_query)
        )
        try:
            await self._shutdown.wait()
        finally:
            poll_task.cancel()
            if self.scheduler:
                self.scheduler.shutdown()
            await self.tg.close()

    async def stop(self) -> None:
        self._shutdown.set()

    # ---- callbacks ----

    async def _fire(self, channel_id: str, trigger: str) -> None:
        assert self.orchestrator is not None
        await self.orchestrator.fire_channel(channel_id, triggered_by=trigger)

    async def _on_message(self, msg: dict) -> None:
        assert self.orchestrator is not None and self.feedback is not None
        assert self.channel_repo is not None and self.onboarding is not None
        text: str = msg.get("text") or ""
        if not text:
            return
        locale_hint: str | None = (msg.get("from") or {}).get("language_code")

        # 1) commands always win (allows /cancel mid-flow, etc.)
        if text.startswith("/"):
            await self._handle_command(text, locale_hint=locale_hint)
            return

        # 2) onboarding flow (waiting for the user to describe a channel)
        if await self.onboarding.on_user_text(text, user_locale_hint=locale_hint):
            return

        # 3) news-Q&A flow (waiting for a question about a post)
        if self.news_qa and await self.news_qa.on_user_text(text):
            return

        # 4) feedback-reply path
        if await self.feedback.on_user_text(text):
            return

        # 4) otherwise ignore in v1
        log.info("Ignoring free-text message: %r", text[:80])

    async def _on_callback_query(self, cq: dict) -> None:
        assert self.feedback is not None and self.orchestrator is not None
        assert self.onboarding is not None
        cq_id = cq["id"]
        data = cq.get("data", "")
        try:
            if data == "setup:approve":
                await self.onboarding.on_approve()
            elif data == "setup:more":
                await self.onboarding.on_more_feedback()
            elif data == "setup:cancel":
                await self.onboarding.cancel()
            elif data.startswith("sources:"):
                msg_id = int(data.split(":", 1)[1])
                await self._show_sources(msg_id)
            elif data.startswith("feedback:"):
                _, channel_id, mid = data.split(":", 2)
                await self.feedback.on_feedback_button(channel_id, int(mid))
            elif data.startswith("prompt:approve:"):
                await self.feedback.on_approve(int(data.split(":", 2)[2]))
            elif data.startswith("prompt:cancel:"):
                await self.feedback.on_cancel(int(data.split(":", 2)[2]))
            elif data.startswith("prompt:more:"):
                await self.feedback.on_more_feedback(int(data.split(":", 2)[2]))
            elif data.startswith("now:"):
                channel_id = data.split(":", 1)[1]
                await self.tg.send_message(f"Firing `{channel_id}`…")
                await self.orchestrator.fire_channel(channel_id, triggered_by="/now")
            elif data.startswith("del:confirm:"):
                channel_id = data.split(":", 2)[2]
                await self._delete_channel(channel_id)
            elif data.startswith("del:cancel:"):
                await self.tg.send_message("Cancelled.")
            elif data.startswith("pause:"):
                channel_id = data.split(":", 1)[1]
                await self.channel_repo.set_enabled(channel_id, False)
                if self.scheduler:
                    self.scheduler.unschedule_channel(channel_id)
                await self.tg.send_message(f"Paused `{channel_id}`")
            elif data.startswith("resume:"):
                channel_id = data.split(":", 1)[1]
                await self.channel_repo.set_enabled(channel_id, True)
                if self.scheduler:
                    await self.scheduler.schedule_channel(channel_id)
                await self.tg.send_message(f"Resumed `{channel_id}`")
            elif data.startswith("edit:"):
                channel_id = data.split(":", 1)[1]
                locale_hint = (cq.get("from") or {}).get("language_code")
                await self.onboarding.start_edit(
                    channel_id, user_locale_hint=locale_hint
                )
            elif data == "ask:done":
                if self.news_qa:
                    await self.news_qa.on_done()
            elif data.startswith("ask:"):
                _, channel_id, mid = data.split(":", 2)
                if self.news_qa:
                    await self.news_qa.on_ask_button(channel_id, int(mid))
        finally:
            await self.tg.answer_callback_query(cq_id)

    async def _show_sources(self, db_message_id: int) -> None:
        assert self.message_repo is not None
        msg = await self.message_repo.get(db_message_id)
        if msg is None or not msg.source_urls:
            await self.tg.send_message("_No sources recorded for that message._")
            return
        lines = [f"{i+1}. {u}" for i, u in enumerate(msg.source_urls)]
        await self.tg.send_message("*Sources:*\n" + "\n".join(lines))

    async def _handle_command(self, text: str, locale_hint: str | None = None) -> None:
        assert self.orchestrator is not None and self.channel_repo is not None
        assert self.onboarding is not None
        first_line, _, rest = text.partition("\n")
        parts = first_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd == "/start":
            await self._send_welcome()
        elif cmd == "/cancel":
            if self.onboarding.is_active():
                await self.onboarding.cancel()
            else:
                await self.tg.send_message("_Nothing to cancel._")
        elif cmd == "/now":
            if args:
                await self.orchestrator.fire_channel(args[0], triggered_by="/now")
            else:
                await self._send_channel_picker("now", "Pick a channel to fire now:")
        elif cmd == "/channels":
            await self._send_channels_overview()
        elif cmd == "/pause":
            if args:
                await self.channel_repo.set_enabled(args[0], False)
                if self.scheduler:
                    self.scheduler.unschedule_channel(args[0])
                await self.tg.send_message(f"Paused `{args[0]}`")
            else:
                await self._send_channel_picker("pause", "Pick a channel to pause:")
        elif cmd == "/resume":
            if args:
                await self.channel_repo.set_enabled(args[0], True)
                if self.scheduler:
                    await self.scheduler.schedule_channel(args[0])
                await self.tg.send_message(f"Resumed `{args[0]}`")
            else:
                await self._send_channel_picker("resume", "Pick a channel to resume:")
        elif cmd in ("/del_channel", "/delchannel"):
            if args:
                await self._confirm_delete_channel(args[0])
            else:
                await self._send_channel_picker(
                    "del:confirm", "Pick a channel to delete (this is permanent):"
                )
        elif cmd in ("/create_channel", "/addchannel"):
            await self.onboarding.start(user_locale_hint=locale_hint)
        elif cmd in ("/change_channel", "/changechannel", "/editchannel"):
            if args:
                await self.onboarding.start_edit(args[0], user_locale_hint=locale_hint)
            else:
                await self._send_channel_picker("edit", "Pick a channel to change:")
        elif cmd == "/usage":
            usage_repo = UsageRepo(self.message_repo.db)  # type: ignore[attr-defined]
            rows = await usage_repo.today_breakdown()
            if not rows:
                await self.tg.send_message("_No usage in the last 24h._")
                return
            lines = [
                f"`{r['channel_id'] or 'system'}` · {r['agent']} · {r['model']} · "
                f"in={r['tin']} out={r['tout']} ${r['cost']:.4f}"
                for r in rows
            ]
            await self.tg.send_message("Usage (last 24h):\n" + "\n".join(lines))
        elif cmd == "/help":
            await self._send_help()
        else:
            await self.tg.send_message(f"Unknown command: `{cmd}`. Try /help.")

    async def _send_welcome(self) -> None:
        await self.tg.send_message(
            "*Your personal news bot* 📰\n\n"
            "I'm a private news assistant. You tell me what topics you care about — "
            "AI, geopolitics, your favourite football club, a niche research area — "
            "and I quietly research the web on a schedule and drop short, "
            "well-sourced briefings right here in this chat.\n\n"
            "Every news post comes with a *🔗 Sources* button so you can verify "
            "anything I claim, and a *✏️ Feedback* button so you can tell me "
            '_"shorter"_, _"more technical"_, _"skip funding rounds"_ — and I\'ll '
            "rewrite the channel's prompt for you.\n\n"
            "*Not just news.* The same engine works for any recurring drip you'd "
            "like: a *Kazakh word a day*, a *daily interesting fact*, a *math "
            "concept of the morning*, an *English idiom*, a *historical event on "
            "this date*. If it can be described, it can be scheduled.\n\n"
            "*Commands*\n"
            "• /create_channel — set up a new feed by *describing* what you want\n"
            "• /channels — list your channels (pause / fire / delete buttons)\n"
            "• /now — fire a channel right now\n"
            "• /pause, /resume, /del_channel — manage a channel\n"
            "• /usage — last 24h spend\n"
            "• /help — show this again\n\n"
            "*Get started:* tap /create_channel and describe — in your own words — "
            "what you want to read, how often, and any style notes."
        )

    async def _send_help(self) -> None:
        await self.tg.send_message(
            "*Commands*\n"
            "/start — welcome + overview\n"
            "/create_channel — guided setup; just describe what you want\n"
            "/change_channel [channel] — change schedule/prompt of an existing channel (no arg → picker)\n"
            "/channels — list channels with action buttons\n"
            "/now [channel] — fire a channel now (no arg → picker)\n"
            "/del_channel [channel] — delete a channel (no arg → picker)\n"
            "/pause [channel] /resume [channel]\n"
            "/usage — last-24h spend breakdown\n"
            "/cancel — abort an in-progress onboarding"
        )

    async def _update_bot_commands(self) -> None:
        assert self.channel_repo is not None
        chs = await self.channel_repo.list_all()
        has_channels = bool(chs)
        tail = [
            ("change_channel", "Change an existing channel's schedule or prompt"),
            ("pause", "Pause a channel"),
            ("resume", "Resume a channel"),
            ("usage", "Show last-24h spend"),
            ("help", "Show command help"),
        ]
        if has_channels:
            commands = [
                ("now", "Fire a channel now (pick from list)"),
                ("channels", "List all channels"),
                ("create_channel", "Add a new channel (guided)"),
                ("del_channel", "Delete a channel"),
                *tail,
                ("start", "Welcome + overview"),
            ]
        else:
            commands = [
                ("start", "Welcome + overview"),
                ("create_channel", "Add a new channel (guided)"),
                ("del_channel", "Delete a channel"),
                *tail,
            ]
        try:
            await self.tg.set_my_commands(commands)
        except Exception as e:
            log.warning("setMyCommands failed: %s", e)

    async def _on_channel_saved(self, channel_id: str) -> None:
        if self.scheduler:
            self.scheduler.unschedule_channel(channel_id)
            await self.scheduler.schedule_channel(channel_id)
        await self._update_bot_commands()

    # ---- channel pickers & mutations ----

    async def _send_channels_overview(self) -> None:
        assert self.channel_repo is not None
        chs = await self.channel_repo.list_all()
        if not chs:
            await self.tg.send_message("_No channels configured. Use /create_channel._")
            return
        lines = [
            f"`{c.id}` — {c.display_name} ({c.mode}) {'🟢' if c.enabled else '🔴'}"
            for c in chs
        ]
        await self.tg.send_message("Channels:\n" + "\n".join(lines))

    async def _send_channel_picker(self, action: str, header: str) -> None:
        assert self.channel_repo is not None
        chs = await self.channel_repo.list_all()
        if not chs:
            await self.tg.send_message("_No channels configured._")
            return
        buttons = [
            InlineButton(
                f"{c.display_name} ({c.id}) {'🟢' if c.enabled else '🔴'}",
                f"{action}:{c.id}",
            )
            for c in chs
        ]
        await self.tg.send_message(header, buttons=buttons, button_columns=1)

    async def _confirm_delete_channel(self, channel_id: str) -> None:
        assert self.channel_repo is not None
        ch = await self.channel_repo.get(channel_id)
        if ch is None:
            await self.tg.send_message(f"Channel `{channel_id}` not found.")
            return
        await self.tg.send_message(
            f"Delete *{ch.display_name}* (`{ch.id}`) permanently?",
            buttons=[
                InlineButton("🗑 Yes, delete", f"del:confirm:{ch.id}"),
                InlineButton("Cancel", f"del:cancel:{ch.id}"),
            ],
            button_columns=2,
        )

    async def _delete_channel(self, channel_id: str) -> None:
        assert self.channel_repo is not None
        ch = await self.channel_repo.get(channel_id)
        if ch is None:
            await self.tg.send_message(f"Channel `{channel_id}` not found.")
            return
        if self.scheduler:
            self.scheduler.unschedule_channel(channel_id)
        await self.channel_repo.delete(channel_id)
        yaml_path = ROOT / "channels" / f"{channel_id}.yaml"
        try:
            if yaml_path.exists():
                yaml_path.unlink()
        except Exception as e:
            log.warning("Failed to remove %s: %s", yaml_path, e)
        await self.tg.send_message(f"Deleted `{channel_id}` ✅")
        await self._update_bot_commands()


async def _amain() -> None:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        sys.stderr.write(
            f"Missing config.yaml. Copy config.example.yaml to {cfg_path} and fill in credentials.\n"
        )
        sys.exit(2)
    cfg = load_config(cfg_path)
    _setup_logging(cfg.logging.level)
    _maybe_init_laminar(cfg)
    app = App(cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(app.stop()))
        except NotImplementedError:
            pass

    await app.start()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
