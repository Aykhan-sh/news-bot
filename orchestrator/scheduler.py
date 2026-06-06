from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from storage.repositories import ChannelRepo, ChannelRow

log = logging.getLogger(__name__)


class ChannelScheduler:
    """Schedules channel firings: cron / interval / probabilistic."""

    def __init__(
        self,
        channels: ChannelRepo,
        fire_callback: Callable[[str, str], Awaitable[None]],
        probabilistic_jitter_seconds: int = 1800,
    ) -> None:
        self.channels = channels
        self.fire = fire_callback
        self.sched = AsyncIOScheduler()
        self.jitter = probabilistic_jitter_seconds

    def start(self) -> None:
        self.sched.start()

    def shutdown(self) -> None:
        self.sched.shutdown(wait=False)

    async def schedule_all(self) -> None:
        for ch in await self.channels.list_enabled():
            self._schedule_one(ch)

    async def schedule_channel(self, channel_id: str) -> None:
        ch = await self.channels.get(channel_id)
        if ch is not None and ch.enabled:
            self._schedule_one(ch)

    def unschedule_channel(self, channel_id: str) -> None:
        for jid in (f"channel:{channel_id}", *[f"channel:{channel_id}:prob:{i}" for i in range(16)]):
            try:
                self.sched.remove_job(jid)
            except Exception:
                pass

    def _schedule_one(self, ch: ChannelRow) -> None:
        spec = ch.schedule_spec
        kind = ch.schedule_kind
        job_id = f"channel:{ch.id}"
        try:
            self.sched.remove_job(job_id)
        except Exception:
            pass
        if kind == "cron":
            if isinstance(spec, str):
                trig = CronTrigger.from_crontab(spec)
            elif isinstance(spec, dict) and isinstance(spec.get("expr"), str):
                trig = CronTrigger.from_crontab(spec["expr"], timezone=spec.get("timezone"))
            elif isinstance(spec, dict):
                allowed_keys = {"year", "month", "day", "week", "day_of_week",
                                "hour", "minute", "second", "timezone"}
                kwargs = {k: v for k, v in spec.items() if k in allowed_keys}
                if not any(k in kwargs for k in ("year", "month", "day", "week",
                                                  "day_of_week", "hour", "minute", "second")):
                    kwargs.setdefault("hour", 9)
                    kwargs.setdefault("minute", 0)
                trig = CronTrigger(**kwargs)
            else:
                trig = CronTrigger.from_crontab("0 9 * * *")
            self.sched.add_job(self._wrapped_fire, trig, args=[ch.id, "cron"], id=job_id, replace_existing=True)
        elif kind == "interval":
            kwargs = spec if isinstance(spec, dict) else {"hours": int(spec)}
            allowed = {k: kwargs[k] for k in ("weeks", "days", "hours", "minutes", "seconds") if k in kwargs}
            if not allowed:
                allowed = {"hours": 6}
            self.sched.add_job(self._wrapped_fire, IntervalTrigger(**allowed), args=[ch.id, "interval"], id=job_id, replace_existing=True)
        elif kind == "probabilistic":
            window = spec if isinstance(spec, dict) else json.loads(spec)
            self._schedule_probabilistic(ch, window)
        else:
            log.warning("Unknown schedule_kind=%s for %s", kind, ch.id)

    def _schedule_probabilistic(self, ch: ChannelRow, window: dict) -> None:
        """Fire roughly `times_per_day` (or `per_day`) random moments within [start_hour, end_hour]."""
        times = int(window.get("times_per_day", window.get("per_day", 2)))
        start_h = int(window.get("start_hour", 9))
        end_h = int(window.get("end_hour", 22))
        tz = window.get("timezone")
        for i in range(times):
            mins = random.randint(start_h * 60, max(start_h * 60 + 1, end_h * 60))
            hour, minute = divmod(mins, 60)
            jitter = random.randint(-self.jitter, self.jitter) // 60
            minute = max(0, min(59, minute + jitter))
            trig_kwargs: dict = {"hour": hour, "minute": minute}
            if tz:
                trig_kwargs["timezone"] = tz
            trig = CronTrigger(**trig_kwargs)
            job_id = f"channel:{ch.id}:prob:{i}"
            self.sched.add_job(self._wrapped_fire, trig, args=[ch.id, "probabilistic"], id=job_id, replace_existing=True)

    async def _wrapped_fire(self, channel_id: str, trigger: str) -> None:
        try:
            await self.fire(channel_id, trigger)
        except Exception as e:
            log.exception("Channel %s firing raised: %s", channel_id, e)
