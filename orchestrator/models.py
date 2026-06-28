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


class ProviderConfig(BaseModel):
    """Credentials for a single pydantic-ai provider.

    The map key in `AppConfig.providers` is the provider name (e.g. `openai`,
    `fireworks`, `together`). These values are exported as `{NAME}_API_KEY` /
    `{NAME}_BASE_URL` environment variables so pydantic-ai's `infer_model`
    resolves them automatically — no provider is hardcoded in Python.
    """

    api_key: str = ""
    base_url: Optional[str] = None


class ModelsConfig(BaseModel):
    """Per-role model identifiers in pydantic-ai `provider:model` form.

    Examples: `openai-responses:gpt-5.4-mini` (OpenAI Responses API — enables
    native web search), `openai-chat:gpt-5.4-mini`,
    `fireworks:accounts/fireworks/models/kimi-k2.6`, `groq:llama-3.3-70b`.
    """

    writer: str = "openai-responses:gpt-5.4-mini"
    researcher: str = "openai-responses:gpt-5.4-mini"
    refiner: str = "openai-responses:gpt-5.4-mini"
    setup: str = "openai-responses:gpt-5.4-mini"
    # Embeddings stay on OpenAI (pydantic-ai does not cover embeddings). May be a
    # bare model name or an `openai:`-prefixed string.
    embedding: str = "text-embedding-3-small"
    # Apply OpenAI's "flex" service tier to `openai-responses:` models.
    openai_flex: bool = False


class ResearcherConfig(BaseModel):
    per_tick_fetch_budget: int = 8
    # Hard cap on `check_relevance` (collision-check) calls per channel tick. Lets
    # the researcher fetch a batch, check it, and — if everything collides — refine
    # its queries and try again, without looping forever.
    per_tick_check_budget: int = 3
    # Deep-research mode only: the most supporting sources the researcher may attach
    # to the anchor story (beyond the anchor itself) before handing them to the
    # writer. The agent is free to pick fewer. Ignored for `single` channels.
    deep_max_sources: int = 4
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


class StorageConfig(BaseModel):
    db_path: str = "data/news-bot.sqlite"


class ObservabilityConfig(BaseModel):
    lmnr_enabled: bool = False
    lmnr_project_api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppConfig(BaseModel):
    telegram: TelegramConfig
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    researcher: ResearcherConfig = Field(default_factory=ResearcherConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def model_for(self, role: str) -> str:
        """Return the configured `provider:model` string for a role.

        `role` is one of: writer, researcher, refiner, setup.
        """
        return getattr(self.models, role)

    def openai_api_key(self) -> str:
        """API key for the `openai` provider, used for embeddings."""
        p = self.providers.get("openai")
        return p.api_key if p else ""

    def embedding_model(self) -> str:
        """Bare embedding model name (strips any `provider:` prefix)."""
        m = self.models.embedding
        return m.split(":", 1)[1] if ":" in m else m


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
