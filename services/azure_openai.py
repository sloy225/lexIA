"""
AI client for Azure AI Foundry OpenAI-compatible endpoints.

Azure AI Foundry project endpoints expose an OpenAI v1-compatible API at:
  https://<hub>.services.ai.azure.com/api/projects/<project>/openai/v1

The plain `openai.OpenAI` client with a custom `base_url` is the correct
SDK for this path — the `azure-ai-inference` SDK appends `?api-version=`
which the /v1 path rejects.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


def _normalize_endpoint(endpoint: str) -> str:
    """
    Return the base URL expected by the OpenAI client:
      https://<hub>.services.ai.azure.com/api/projects/<proj>/openai/v1

    Strips any trailing operation path (/responses, /chat/completions, etc.)
    and ensures there is no trailing slash.
    """
    # Remove everything after /v1 (e.g. /responses, /chat/completions)
    endpoint = re.sub(r"(/v1)/.*$", r"\1", endpoint.rstrip("/"))
    return endpoint


class AzureOpenAIService:
    """
    Thin wrapper around an Azure AI Foundry OpenAI-compatible endpoint.
    Uses the `openai` SDK with a custom base_url — no api-version needed.
    """

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            base_url = _normalize_endpoint(settings.AZURE_AI_FOUNDRY_ENDPOINT)
            logger.info("Connecting to AI Foundry endpoint: %s", base_url)
            self._client = OpenAI(
                base_url=base_url,
                api_key=settings.AZURE_AI_FOUNDRY_API_KEY,
            )
        return self._client

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=settings.AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT,
            input=text,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
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
        response = self.client.chat.completions.create(
            model=settings.AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            return self._stream_response(response)

        return response.choices[0].message.content or ""

    def _stream_response(self, response) -> Iterator[str]:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
