from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from orchestrator.models import CostControlConfig
from storage.repositories import UsageRepo

log = logging.getLogger(__name__)


@dataclass
class PrecheckResult:
    allowed: bool
    model_override: Optional[str] = None
    reason: str = ""
    warn: bool = False


class CostController:
    def __init__(self, usage: UsageRepo, cfg: CostControlConfig) -> None:
        self.usage = usage
        self.cfg = cfg

    async def precheck(self, channel_id: Optional[str], requested_model: str) -> PrecheckResult:
        global_today = await self.usage.today_total(None)
        if global_today >= self.cfg.global_daily_usd:
            return self._on_threshold(
                requested_model,
                f"global budget exhausted (${global_today:.4f} / ${self.cfg.global_daily_usd:.2f})",
            )
        if channel_id is not None:
            chan_today = await self.usage.today_total(channel_id)
            if chan_today >= self.cfg.per_channel_default_daily_usd:
                return self._on_threshold(
                    requested_model,
                    f"channel '{channel_id}' budget exhausted (${chan_today:.4f})",
                )
        return PrecheckResult(allowed=True, model_override=None, reason="ok")

    def _on_threshold(self, requested_model: str, reason: str) -> PrecheckResult:
        if self.cfg.on_threshold == "warn":
            return PrecheckResult(allowed=True, reason=reason, warn=True)
        if self.cfg.on_threshold == "downgrade":
            return PrecheckResult(
                allowed=True,
                model_override=self.cfg.downgrade_to,
                reason=f"downgrade: {reason}",
                warn=True,
            )
        return PrecheckResult(allowed=False, reason=reason)
