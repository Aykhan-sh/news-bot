from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiohttp

from orchestrator.models import TelegramConfig

log = logging.getLogger(__name__)


@dataclass
class InlineButton:
    text: str
    callback_data: str


class TelegramBotClient:
    """Minimal async Bot API client.

    Long-polls `getUpdates` and dispatches:
      - text messages (commands + free-text replies) → on_message
      - inline-button taps → on_callback_query

    Keeps it tiny on purpose; this is the only Telegram surface in v1.
    """

    def __init__(self, cfg: TelegramConfig, session_dir: str | Path = "data") -> None:
        self.cfg = cfg
        self.base = f"https://api.telegram.org/bot{cfg.bot_token}"
        self._offset_path = Path(session_dir) / "tg_offset.json"
        self._offset_path.parent.mkdir(parents=True, exist_ok=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._stopped = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        self._stopped = True
        if self._session and not self._session.closed:
            await self._session.close()

    async def _api(self, method: str, payload: dict | None = None) -> dict:
        sess = await self._ensure_session()
        url = f"{self.base}/{method}"
        async with sess.post(url, json=payload or {}) as resp:
            data = await resp.json()
        if not data.get("ok"):
            log.warning("Telegram %s failed: %s", method, data)
        return data

    # ---- read offset ----

    def _load_offset(self) -> int:
        try:
            if self._offset_path.exists():
                return int(json.loads(self._offset_path.read_text())["offset"])
        except Exception:
            pass
        return 0

    def _save_offset(self, offset: int) -> None:
        try:
            self._offset_path.write_text(json.dumps({"offset": offset}))
        except Exception as e:
            log.warning("Failed to save tg offset: %s", e)

    # ---- send ----

    async def set_my_commands(self, commands: list[tuple[str, str]]) -> None:
        await self._api("setMyCommands", {
            "commands": [{"command": c, "description": d} for c, d in commands],
        })

    # Telegram hard-caps a text message at 4096 chars; stay safely under it.
    _MAX_TEXT = 4000

    _TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)[^>]*>")

    @classmethod
    def _balance_tags(cls, chunk: str) -> tuple[str, str]:
        """Return (closers, openers) to balance a chunk's HTML tags.

        Telegram rejects a message whose entities aren't closed within it, so
        when a chunk is cut mid-`<pre>`/`<b>`/… we must close the open tags at
        the end of this chunk and re-open them at the start of the next. Returns
        the closing tags to append here and the original opening tags to prepend
        to the remainder.
        """
        stack: list[tuple[str, str]] = []  # (tag_name, full_opening_tag)
        for m in cls._TAG_RE.finditer(chunk):
            name = m.group(2).lower()
            if m.group(1) == "/":
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == name:
                        del stack[i]
                        break
            else:
                stack.append((name, m.group(0)))
        closers = "".join(f"</{name}>" for name, _ in reversed(stack))
        openers = "".join(full for _, full in stack)
        return closers, openers

    @classmethod
    def _split_text(cls, text: str) -> list[str]:
        """Split text into Telegram-sized chunks, preferring newline boundaries
        and keeping HTML tags balanced within each chunk."""
        if len(text) <= cls._MAX_TEXT:
            return [text]
        chunks: list[str] = []
        remaining = text
        carry = ""
        while remaining:
            remaining = carry + remaining
            carry = ""
            if len(remaining) <= cls._MAX_TEXT:
                chunks.append(remaining)
                break
            window = remaining[: cls._MAX_TEXT]
            split_at = window.rfind("\n")
            if split_at <= 0:
                # No newline to break on; avoid cutting inside a tag.
                last_gt = window.rfind(">")
                last_lt = window.rfind("<")
                split_at = (
                    last_lt if last_lt > last_gt else cls._MAX_TEXT
                )
                if split_at <= 0:
                    split_at = cls._MAX_TEXT
            chunk = remaining[:split_at]
            rest = remaining[split_at:].lstrip("\n")
            closers, openers = cls._balance_tags(chunk)
            chunks.append(chunk + closers)
            carry = openers
            remaining = rest
        return chunks

    async def _send_one(self, payload: dict[str, Any], parse_mode: Optional[str]) -> Optional[int]:
        data = await self._api("sendMessage", payload)
        if data.get("ok"):
            return data["result"]["message_id"]
        if parse_mode and "parse" in str(data.get("description", "")).lower():
            log.warning("Parse mode %s rejected by Telegram (%s); sending as plain text", parse_mode, data.get("description"))
            retry = dict(payload)
            retry.pop("parse_mode", None)
            data = await self._api("sendMessage", retry)
            if data.get("ok"):
                return data["result"]["message_id"]
        return None

    async def send_message(
        self,
        text: str,
        *,
        chat_id: Optional[int] = None,
        reply_to: Optional[int] = None,
        buttons: Optional[list[InlineButton]] = None,
        button_columns: int = 1,
        parse_mode: Optional[str] = "HTML",
        force_reply: bool = False,
    ) -> Optional[int]:
        chat = chat_id or self.cfg.owner_chat_id
        reply_markup: Optional[dict[str, Any]] = None
        if buttons:
            cols = max(1, button_columns)
            rows = [buttons[i:i + cols] for i in range(0, len(buttons), cols)]
            reply_markup = {
                "inline_keyboard": [
                    [{"text": b.text, "callback_data": b.callback_data} for b in row]
                    for row in rows
                ]
            }
        elif force_reply:
            reply_markup = {"force_reply": True, "selective": False}

        chunks = self._split_text(text)
        last_id: Optional[int] = None
        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            payload: dict[str, Any] = {
                "chat_id": chat,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if is_last and reply_to:
                payload["reply_to_message_id"] = reply_to
            if is_last and reply_markup is not None:
                payload["reply_markup"] = reply_markup
            last_id = await self._send_one(payload, parse_mode)
        return last_id

    async def edit_reply_markup(self, message_id: int, buttons: Optional[list[InlineButton]] = None) -> None:
        payload: dict[str, Any] = {
            "chat_id": self.cfg.owner_chat_id,
            "message_id": message_id,
        }
        if buttons is None:
            payload["reply_markup"] = {"inline_keyboard": []}
        else:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [{"text": b.text, "callback_data": b.callback_data} for b in buttons]
                ]
            }
        await self._api("editMessageReplyMarkup", payload)

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        await self._api("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text,
        })

    # ---- poll loop ----

    async def poll(
        self,
        on_message: Callable[[dict], Awaitable[None]],
        on_callback_query: Callable[[dict], Awaitable[None]],
    ) -> None:
        offset = self._load_offset()
        while not self._stopped:
            payload = {"timeout": 25, "offset": offset, "allowed_updates": ["message", "callback_query"]}
            try:
                data = await self._api("getUpdates", payload)
            except Exception as e:
                log.warning("getUpdates failed: %s", e)
                await asyncio.sleep(5)
                continue
            updates = data.get("result", []) if isinstance(data, dict) else []
            for upd in updates:
                offset = max(offset, upd["update_id"] + 1)
                self._save_offset(offset)
                if upd.get("message"):
                    msg = upd["message"]
                    if msg.get("chat", {}).get("id") != self.cfg.owner_chat_id:
                        continue
                    try:
                        await on_message(msg)
                    except Exception as e:
                        log.exception("on_message handler raised: %s", e)
                elif upd.get("callback_query"):
                    cq = upd["callback_query"]
                    if cq.get("from", {}).get("id") != self.cfg.owner_chat_id:
                        continue
                    try:
                        await on_callback_query(cq)
                    except Exception as e:
                        log.exception("on_callback_query handler raised: %s", e)
