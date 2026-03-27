"""
AI client backed by Azure AI Foundry via the azure-ai-inference SDK.

The public interface (embed / embed_batch / chat / chat_with_system) is
identical to the previous implementation so no other module needs changes.
"""
from __future__ import annotations

import logging
from typing import Iterator

from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatRequestMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from config.settings import settings

logger = logging.getLogger(__name__)


def _build_messages(raw: list[dict]) -> list[ChatRequestMessage]:
    """Convert plain dicts to azure-ai-inference message objects."""
    result = []
    for m in raw:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AssistantMessage(content=content))
        else:
            result.append(UserMessage(content=content))
    return result


class AzureOpenAIService:
    """
    Thin wrapper around Azure AI Foundry inference endpoints.
    Uses ChatCompletionsClient for generation and EmbeddingsClient for vectors.
    """

    def __init__(self) -> None:
        self._credential = AzureKeyCredential(settings.AZURE_AI_FOUNDRY_API_KEY)
        self._chat_client: ChatCompletionsClient | None = None
        self._embed_client: EmbeddingsClient | None = None

    @property
    def chat_client(self) -> ChatCompletionsClient:
        if self._chat_client is None:
            self._chat_client = ChatCompletionsClient(
                endpoint=settings.AZURE_AI_FOUNDRY_ENDPOINT,
                credential=self._credential,
            )
        return self._chat_client

    @property
    def embed_client(self) -> EmbeddingsClient:
        if self._embed_client is None:
            self._embed_client = EmbeddingsClient(
                endpoint=settings.AZURE_AI_FOUNDRY_ENDPOINT,
                credential=self._credential,
            )
        return self._embed_client

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        response = self.embed_client.embed(
            model=settings.AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT,
            input=[text],
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        response = self.embed_client.embed(
            model=settings.AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT,
            input=texts,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Send a chat completion request via Azure AI Foundry.
        If stream=True, yields text chunks; otherwise returns the full string.
        """
        ai_messages = _build_messages(messages)

        response = self.chat_client.complete(
            model=settings.AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT,
            messages=ai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            return self._stream_response(response)

        return response.choices[0].message.content or ""

    def _stream_response(self, response) -> Iterator[str]:
        for update in response:
            if update.choices and update.choices[0].delta.content:
                yield update.choices[0].delta.content

    def chat_with_system(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)


azure_openai_service = AzureOpenAIService()
