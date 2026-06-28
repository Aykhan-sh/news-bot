from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers import infer_provider
from pydantic_ai.settings import ModelSettings

from orchestrator.models import ModelsConfig, ProviderConfig

log = logging.getLogger(__name__)


class _ShortNameModel(WrapperModel):
    """Reports the basename of the wrapped model's id in telemetry.

    Some providers use a namespaced model id (e.g. Fireworks'
    `accounts/fireworks/models/minimax-m3`). PydanticAI puts that full id on the
    `gen_ai.request.model` / `gen_ai.response.model` span attributes, which
    Laminar can't match to its pricing table, so cost is dropped. This wrapper
    exposes only the last path segment (`minimax-m3`) on the telemetry while the
    wrapped model still sends the full id in the real API request.
    """

    @staticmethod
    def _short(name: Optional[str]) -> Optional[str]:
        return name.rsplit("/", 1)[-1] if name else name

    @property
    def model_name(self) -> str:
        return self._short(self.wrapped.model_name) or ""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        response.model_name = self._short(response.model_name)
        return response


class ModelFactory:
    """Builds PydanticAI models from `provider:model` strings.

    There is no hardcoded provider logic: credentials from `providers` are
    exported to `{NAME}_API_KEY` / `{NAME}_BASE_URL` environment variables and
    `pydantic_ai.models.infer_model` resolves the right provider/model class.
    Any provider pydantic-ai supports (openai, fireworks, together, groq,
    mistral, deepseek, openrouter, ...) works with no code change — just add it
    to config.
    """

    def __init__(
        self, providers: dict[str, ProviderConfig], models: ModelsConfig
    ) -> None:
        self.models = models
        self._export_provider_env(providers)

    @staticmethod
    def _export_provider_env(providers: dict[str, ProviderConfig]) -> None:
        for name, p in providers.items():
            env = name.upper().replace("-", "_")
            if p.api_key:
                os.environ.setdefault(f"{env}_API_KEY", p.api_key)
            if p.base_url:
                os.environ.setdefault(f"{env}_BASE_URL", p.base_url)

    def get(self, model_string: str) -> Model:
        """Resolve a `provider:model` string to a configured PydanticAI model."""
        prefix, _, name = model_string.partition(":")
        if prefix == "openai-responses" and self.models.openai_flex:
            model: Model = OpenAIResponsesModel(
                name,
                provider=infer_provider("openai"),
                settings=OpenAIResponsesModelSettings(openai_service_tier="flex"),
            )
        else:
            model = infer_model(model_string)
        # Namespaced ids (e.g. Fireworks' `accounts/fireworks/models/...`) break
        # Laminar's cost lookup; expose only the basename in telemetry.
        if "/" in model.model_name:
            model = _ShortNameModel(model)
        return model

    @staticmethod
    def supports_native_web_search(model: Optional[Model]) -> bool:
        """Whether `model` exposes a provider-native web-search built-in.

        Only OpenAI's Responses API (`OpenAIResponsesModel`) supports the
        built-in `WebSearchTool`. Everything else must use the researcher's
        DuckDuckGo-backed fallback `web_search` function tool.
        """
        while isinstance(model, WrapperModel):
            model = model.wrapped
        return isinstance(model, OpenAIResponsesModel)
