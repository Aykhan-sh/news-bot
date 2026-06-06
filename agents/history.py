from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


def build_message_history(
    system_prompt: str, turns: list[dict]
) -> list[ModelMessage]:
    """Build a pydantic-ai message history for a conversational agent.

    The stable `system_prompt` goes in the very first request so it forms a
    cache-friendly prefix that never changes between turns, and the prior
    `turns` follow as alternating request/response messages — keeping the
    agent <-> user exchange in the conversation, not the system prompt. The
    caller passes the newest user message separately as `user_prompt`.

    Each turn is a mapping `{"role": "user" | "assistant", "text": str}`.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
    ]
    for turn in turns:
        text = turn["text"]
        if turn["role"] == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=text)]))
    return messages
