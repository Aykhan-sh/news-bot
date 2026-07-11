from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from storage.repositories import ChannelRepo, ChannelRow

log = logging.getLogger(__name__)


class ChannelScheduler:
    """Schedules channel firings: cron / interval / probabilistic."""

    def __init__(
        self,
        channels: ChannelRepo,
        fire_callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self.channels = channels
        self.fire = fire_callback
        self.sched = AsyncIOScheduler()

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
        jids = (
            f"channel:{channel_id}",
            f"channel:{channel_id}:prob:planner",
            *[f"channel:{channel_id}:prob:{i}" for i in range(64)],
        )
        for jid in jids:
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
        """Fire roughly `times_per_day` (or `per_day`) random moments within [start_hour, end_hour].

        The firing times are re-randomized *every day* rather than picked once at
        startup: a midnight planner job (in the channel's timezone) chooses fresh
        random one-off firings for that day. Otherwise a long-running process would
        fire at the same frozen times every day.
        """
        times = int(window.get("times_per_day", window.get("per_day", 2)))
        start_h = int(window.get("start_hour", 9))
        end_h = int(window.get("end_hour", 22))
        tz = window.get("timezone")

        planner_kwargs: dict = {"hour": 0, "minute": 0}
        if tz:
            planner_kwargs["timezone"] = tz
        self.sched.add_job(
            self._plan_probabilistic_day,
            CronTrigger(**planner_kwargs),
            args=[ch.id, times, start_h, end_h, tz],
            id=f"channel:{ch.id}:prob:planner",
            replace_existing=True,
        )
        # Populate the remainder of *today* immediately (the planner only runs at
        # midnight, so without this a fresh start would post nothing until tomorrow).
        self._plan_probabilistic_day(ch.id, times, start_h, end_h, tz)

    def _plan_probabilistic_day(
        self, channel_id: str, times: int, start_h: int, end_h: int, tz
    ) -> None:
        """Pick fresh random firing times for the current day and schedule them."""
        tzinfo = ZoneInfo(tz) if tz else self.sched.timezone
        now = datetime.now(tzinfo)
        for i in range(64):
            try:
                self.sched.remove_job(f"channel:{channel_id}:prob:{i}")
            except Exception:
                pass
        scheduled = 0
        for i in range(times):
            mins = random.randint(start_h * 60, max(start_h * 60 + 1, end_h * 60))
            hour, minute = divmod(mins, 60)
            run_at = now.replace(
                hour=hour, minute=minute, second=random.randint(0, 59), microsecond=0
            )
            if run_at <= now:
                continue
            self.sched.add_job(
                self._wrapped_fire,
                DateTrigger(run_date=run_at),
                args=[channel_id, "probabilistic"],
                id=f"channel:{channel_id}:prob:{i}",
                replace_existing=True,
            )
            scheduled += 1
        log.info(
            "Planned %d/%d probabilistic firing(s) for %s today (window %02d:00-%02d:00)",
            scheduled, times, channel_id, start_h, end_h,
        )

    async def _wrapped_fire(self, channel_id: str, trigger: str) -> None:
        try:
            await self.fire(channel_id, trigger)
        except Exception as e:
            log.exception("Channel %s firing raised: %s", channel_id, e)
