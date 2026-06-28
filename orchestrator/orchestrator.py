from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from typing import Optional

from pydantic_ai.settings import ModelSettings

try:  # Laminar is optional; tracing degrades to a no-op when it's absent.
    from lmnr import Laminar
except Exception:  # pragma: no cover - lmnr not installed
    Laminar = None  # type: ignore[assignment]

from agents.deps import ResearcherDeps, WriterDeps
from agents.researcher import (
    ResearcherOutput,
    researcher_agent,
    researcher_agent_fallback,
)
from agents.writer import WriterOutput, writer_agent
from llm.models import ModelFactory
from orchestrator.dedup import DedupEngine, build_embed_text
from orchestrator.models import AppConfig
from orchestrator.prompt_builder import render
from storage.repositories import (
    ChannelRepo,
    ChannelRow,
    MessageRepo,
)
from telegram_client.client import InlineButton, TelegramBotClient

log = logging.getLogger(__name__)


def _tick_span(name: str, **attrs):
    """Group every agent run of one channel tick under a single Laminar trace.

    pydantic-ai's per-agent instrumentation opens a fresh root span (hence a new
    trace) for each `agent.run`, so the researcher and writer of one tick would
    otherwise land in two separate traces. Opening a parent span here makes them
    children of one trace. No-op when Laminar isn't initialised.
    """
    if Laminar is not None and Laminar.is_initialized():
        return Laminar.start_as_current_span(name=name, input=attrs or None)
    return nullcontext()


def _html_escape(text: str) -> str:
    """Escape HTML special characters (& < >) for safe inclusion in Telegram HTML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters that break Telegram's parser.

    Keeps normal whitespace (newline, carriage return, tab) and all printable
    Unicode. Replaces other C0/C1 control chars with a space so words don't run
    together.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in (0x0A, 0x0D, 0x09):
            out.append(ch)
        elif cp < 0x20 or cp == 0x7F or 0x80 <= cp <= 0x9F:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


_MD_HTML_RE = re.compile(
    r"```([\s\S]+?)```"                          # ```code block```
    r"|\*\*([^*\n]+)\*\*"                        # **bold**
    r"|`([^`\n]+)`"                              # `inline code`
    r"|\[([^\]\n]+)\]\((https?://[^)\s\n]+)\)"  # [label](url)
    r"|(?<!\w)\*([^*\n]+)\*(?!\w)"               # *italic* (not inside a word)
    r"|(?<!\w)_([^_\n]+)_(?!\w)"                 # _italic_ (not inside a word)
)


def md_to_html(text: str) -> str:
    """Convert simple Markdown to Telegram HTML.

    Recognises **bold**, *italic*, _italic_, `code`, ```code block```,
    and [label](url). All other text is HTML-escaped so it displays literally.
    """
    text = _strip_control_chars(text)
    result: list[str] = []
    last = 0
    for m in _MD_HTML_RE.finditer(text):
        result.append(_html_escape(text[last:m.start()]))
        if m.group(1) is not None:
            result.append(f"<pre>{_html_escape(m.group(1).strip())}</pre>")
        elif m.group(2) is not None:
            result.append(f"<b>{_html_escape(m.group(2))}</b>")
        elif m.group(3) is not None:
            result.append(f"<code>{_html_escape(m.group(3))}</code>")
        elif m.group(4) is not None:
            label = _html_escape(m.group(4))
            url = m.group(5).replace("&", "&amp;")
            result.append(f'<a href="{url}">{label}</a>')
        elif m.group(6) is not None:
            result.append(f"<i>{_html_escape(m.group(6))}</i>")
        elif m.group(7) is not None:
            result.append(f"<i>{_html_escape(m.group(7))}</i>")
        last = m.end()
    result.append(_html_escape(text[last:]))
    return "".join(result)


class Orchestrator:
    def __init__(
        self,
        cfg: AppConfig,
        tg: TelegramBotClient,
        models: ModelFactory,
        channels: ChannelRepo,
        messages: MessageRepo,
        dedup: DedupEngine,
    ) -> None:
        self.cfg = cfg
        self.tg = tg
        self.models = models
        self.channels = channels
        self.messages = messages
        self.dedup = dedup

    async def fire_channel(self, channel_id: str, triggered_by: str = "scheduler") -> None:
        channel = await self.channels.get(channel_id)
        if channel is None or not channel.enabled:
            log.info("Channel %s not active; skip", channel_id)
            return

        log.info("Firing channel %s (mode=%s, trigger=%s)", channel.id, channel.mode, triggered_by)

        with _tick_span(
            f"channel_tick:{channel.id}",
            channel_id=channel.id,
            mode=channel.mode,
            triggered_by=triggered_by,
        ):
            writer_model = self.cfg.model_for("writer")
            researcher_model = self.cfg.model_for("researcher")

            window = await self.dedup.sliding_window(channel.id, channel.dedup_window_n)

            research_note: Optional[dict] = None
            research_vector: Optional[list[float]] = None
            supporting_notes: list[dict] = []
            if channel.mode == "sourced":
                researcher_result = await self._run_researcher(channel, window, researcher_model)
                if researcher_result is None:
                    await self._send_nothing_new(channel)
                    return
                research_note, research_vector, supporting_notes = researcher_result

            await self._run_writer_and_send(
                channel, window, research_note, research_vector, supporting_notes, writer_model
            )

    # ------------------------------------------------------------------
    # Researcher path
    # ------------------------------------------------------------------

    async def _run_researcher(
        self,
        channel: ChannelRow,
        window,
        researcher_model: str,
    ) -> Optional[tuple[dict, Optional[list[float]], list[dict]]]:
        deps = ResearcherDeps(
            channel=channel,
            messages=self.messages,
            dedup=self.dedup,
            window=window,
            fetch_budget=self.cfg.researcher.per_tick_fetch_budget,
            check_budget=self.cfg.researcher.per_tick_check_budget,
            deep_max_sources=self.cfg.researcher.deep_max_sources,
        )
        model = self.models.get(researcher_model)
        # OpenAI uses the native WebSearchTool; providers without it (Fireworks/GLM)
        # use the fallback agent that exposes a DuckDuckGo-backed web_search tool.
        agent = (
            researcher_agent
            if self.models.supports_native_web_search(model)
            else researcher_agent_fallback
        )
        model_settings: Optional[ModelSettings] = None
        if self.cfg.researcher.temperature is not None:
            # Higher temperature diversifies the search queries across ticks so
            # repeated firings don't keep surfacing the same colliding stories.
            model_settings = ModelSettings(temperature=self.cfg.researcher.temperature)
        user_prompt = render(
            "shared/do_not_repeat.j2",
            window=[
                {
                    "title": m.title,
                    "keywords": m.keywords,
                    "source_urls": m.source_urls,
                    "created_at": m.created_at,
                }
                for m in window
            ],
        ) + "\n\nBegin."
        try:
            result = await agent.run(
                user_prompt=user_prompt,
                deps=deps,
                model=model,
                model_settings=model_settings,
            )
        except Exception as e:
            log.exception("Researcher failed for %s: %s", channel.id, e)
            return None

        out: ResearcherOutput = result.output
        # Deep mode commits the anchor mid-trajectory via the `choose_anchor` tool,
        # which is authoritative; the final output's `picked_id` is the single-mode
        # path and the deep-mode fallback if the tool was never called.
        picked_id = deps.anchor_id or out.picked_id
        if picked_id is None:
            log.info("Researcher returned nothing for %s", channel.id)
            return None

        # The pick is the id of a source the researcher fetched this tick. The full
        # record (url, title, published_at, text) is the writer's research note.
        # We trust check_relevance for freshness/dedup — no extra post-pick gate.
        candidate = deps.fetched.get(picked_id)
        if candidate is None:
            log.warning(
                "Researcher picked id %s for %s but it was never fetched (have: %s); skipping",
                picked_id,
                channel.id,
                list(deps.fetched.keys()),
            )
            return None
        # Deep-research mode: attach the supporting sources the researcher gathered
        # around the anchor. `gather_supporting_sources` already freshness-checked and
        # capped them into deps.supporting_ids (authoritative); fall back to the output
        # field only if the tool was never called. Single-source channels: empty.
        supporting_ids = deps.supporting_ids or out.supporting_ids
        supporting_notes: list[dict] = []
        if supporting_ids:
            seen_ids = {picked_id}
            for sid in supporting_ids:
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                support = deps.fetched.get(sid)
                if support is None:
                    log.warning(
                        "Researcher listed supporting id %s for %s but it was never "
                        "fetched (have: %s); skipping",
                        sid,
                        channel.id,
                        list(deps.fetched.keys()),
                    )
                    continue
                supporting_notes.append(support.to_dict())
                if len(supporting_notes) >= deps.deep_max_sources:
                    break
        log.info(
            "Researcher picked %s (%s) for %s with %d supporting source(s)",
            picked_id,
            out.picked_title or "?",
            channel.id,
            len(supporting_notes),
        )
        picked_vector = deps.fetched_vectors.get(picked_id)
        return candidate.to_dict(), picked_vector, supporting_notes

    async def _send_nothing_new(self, channel: ChannelRow) -> None:
        text = f"{_html_escape(channel.hashtag)}\n\n<i>Nothing new today.</i>"
        await self.tg.send_message(text)

    # ------------------------------------------------------------------
    # Writer path + dedup retry
    # ------------------------------------------------------------------

    async def _run_writer_and_send(
        self,
        channel: ChannelRow,
        window,
        research_note: Optional[dict],
        research_vector: Optional[list[float]],
        supporting_notes: list[dict],
        writer_model: str,
    ) -> None:
        deps = WriterDeps(
            channel=channel,
            window=window,
            research_note=research_note,
            supporting_notes=supporting_notes,
            fetch_budget=self.cfg.researcher.per_tick_fetch_budget,
        )
        model = self.models.get(writer_model)

        # Dedup is the researcher's job (it picks a non-colliding source from a
        # whole shortlist). The writer only has the one chosen source, so re-checking
        # here can only fail — it just writes the post.
        try:
            result = await writer_agent.run(
                user_prompt="Write the next post for this channel.", deps=deps, model=model
            )
        except Exception as e:
            log.exception("Writer failed for %s: %s", channel.id, e)
            return
        draft = result.output

        body_text = self._format_telegram_body(channel, draft)
        buttons = self._buttons_for_draft(channel, draft)
        tg_msg_id = await self.tg.send_message(body_text, buttons=buttons, parse_mode="HTML")

        # Token/cost accounting now lives in Laminar; the messages row keeps zeros.
        msg_id = await self.messages.insert(
            channel_id=channel.id,
            title=draft.title,
            body=draft.body,
            hashtags=draft.hashtags,
            keywords=draft.keywords,
            source_urls=draft.sources_used,
            telegram_message_id=tg_msg_id,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )

        # Store the post's embedding so future ticks can dedup against it.
        # Prefer the researcher's pre-computed candidate vector (already used for
        # collision checking) so the stored signal matches what was actually compared.
        # Fall back to embedding the fetched source text when check_relevance did not
        # store a vector (e.g. parallel tool-call race with final_result). Only use
        # the writer's output as a last resort for unsourced channels.
        try:
            if research_vector is not None:
                vector = research_vector
            elif research_note is not None:
                vector = await self.dedup.embedder.embed(
                    build_embed_text(
                        research_note.get("title") or draft.title,
                        None,
                        research_note.get("text") or None,
                    )
                )
            else:
                vector = await self.dedup.embedder.embed(
                    build_embed_text(draft.title, draft.keywords, draft.body)
                )
            await self.messages.save_embedding(
                msg_id, channel.id, self.dedup.embedder.model, vector
            )
        except Exception as e:
            log.warning("Failed to store embedding for msg %s: %s", msg_id, e)

        # Re-render buttons now that we have a DB message id (so callback payloads can reference it).
        if tg_msg_id is not None:
            buttons_final = self._buttons_for_draft(channel, draft, db_message_id=msg_id)
            try:
                await self.tg.edit_reply_markup(tg_msg_id, buttons=buttons_final)
            except Exception:
                pass

    def _format_telegram_body(self, channel: ChannelRow, draft: WriterOutput) -> str:
        plain_title = re.sub(r"\\([_*\[\]()~`>#+\-=|{}.!])", r"\1", draft.title)
        safe_body = md_to_html(draft.body)
        seen: set[str] = set()
        deduped_hashtags: list[str] = []
        for t in [channel.hashtag] + list(draft.hashtags):
            key = t.lstrip("#").lower()
            if key not in seen:
                seen.add(key)
                deduped_hashtags.append(t)
        hashtag_line = " ".join(
            _html_escape(t if t.startswith("#") else f"#{t.lstrip('#')}")
            for t in deduped_hashtags
        )
        parts = [f"<b>{_html_escape(plain_title)}</b>\n\n{safe_body}\n\n{hashtag_line}"]
        if draft.sources_used:
            src_lines = [
                f"{i+1}. {_html_escape(u)}" for i, u in enumerate(draft.sources_used)
            ]
            parts.append("\n\nsources:\n" + "\n".join(src_lines))
        return "".join(parts)

    def _buttons_for_draft(
        self,
        channel: ChannelRow,
        draft: WriterOutput,
        db_message_id: Optional[int] = None,
    ) -> list[InlineButton]:
        buttons: list[InlineButton] = []
        mid_part = str(db_message_id) if db_message_id is not None else "0"
        buttons.append(InlineButton(text="✏️ Feedback", callback_data=f"feedback:{channel.id}:{mid_part}"))
        buttons.append(InlineButton(text="💬 Ask", callback_data=f"ask:{channel.id}:{mid_part}"))
        return buttons

    async def _notify_owner(self, text: str) -> None:
        try:
            await self.tg.send_message(text)
        except Exception:
            pass
