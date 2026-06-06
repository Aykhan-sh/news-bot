from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class TelegramConfig(BaseModel):
    api_id: int
    api_hash: str
    bot_token: str
    owner_chat_id: int


class OpenAIConfig(BaseModel):
    api_key: str
    default_writer_model: str = "gpt-5.4-mini"
    default_researcher_model: str = "gpt-5.4-mini"
    default_refiner_model: str = "gpt-5.4-mini"
    default_setup_model: str = "gpt-5.4-mini"
    embedding_model: str = "text-embedding-3-small"
    flex_mode: bool = False


class ResearcherConfig(BaseModel):
    per_tick_fetch_budget: int = 8
    # Hard cap on `check_relevance` (collision-check) calls per channel tick. Lets
    # the researcher fetch a batch, check it, and — if everything collides — refine
    # its queries and try again, without looping forever.
    per_tick_check_budget: int = 3
    # Sampling temperature for the researcher's model. Higher values diversify the
    # search queries across ticks so repeated firings don't keep colliding on the
    # same stories. Leave null to use the provider default. NOTE: OpenAI reasoning
    # models (gpt-5.x with reasoning on) reject a non-default temperature — keep
    # this null for those, or point the researcher at a non-reasoning model.
    temperature: Optional[float] = None


class DedupConfig(BaseModel):
    similarity_threshold: float = 0.75
    lookback_days: int = 14
    log_candidates: bool = False
    candidate_log_path: str = "data/dedup_candidates.jsonl"


class CostControlConfig(BaseModel):
    global_daily_usd: float = 1.50
    per_channel_default_daily_usd: float = 0.50
    on_threshold: str = "pause"  # warn | downgrade | pause
    downgrade_to: str = "gpt-5.4-mini"


class SchedulingConfig(BaseModel):
    probabilistic_jitter_seconds: int = 1800


class StorageConfig(BaseModel):
    db_path: str = "data/news-bot.sqlite"


class ObservabilityConfig(BaseModel):
    lmnr_enabled: bool = False
    lmnr_project_api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppConfig(BaseModel):
    telegram: TelegramConfig
    openai: OpenAIConfig
    researcher: ResearcherConfig = Field(default_factory=ResearcherConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> AppConfig:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return AppConfig(**data)


def load_channels_dir(path: str | Path) -> list[dict[str, Any]]:
    """Load every YAML in `channels/` as a dict."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for f in sorted(p.glob("*.yaml")):
        out.append(yaml.safe_load(f.read_text(encoding="utf-8")))
    return out
