from __future__ import annotations

import logging
from typing import Optional

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from orchestrator.models import OpenAIConfig

log = logging.getLogger(__name__)


class ModelFactory:
    """Returns PydanticAI Model instances configured with the user's OpenAI key.

    Uses `OpenAIResponsesModel` (not `OpenAIModel`) so that built-in tools like
    `WebSearchTool` work — the Responses API is the only OpenAI surface that
    exposes server-side `web_search_preview`.
    """

    def __init__(self, cfg: OpenAIConfig) -> None:
        self.cfg = cfg
        self._provider = OpenAIProvider(api_key=cfg.api_key)
        self._settings: OpenAIResponsesModelSettings | None = (
            OpenAIResponsesModelSettings(openai_service_tier="flex") if cfg.flex_mode else None
        )

    def get(self, model_id: Optional[str]) -> OpenAIResponsesModel:
        mid = model_id or self.cfg.default_writer_model
        return OpenAIResponsesModel(mid, provider=self._provider, settings=self._settings)
