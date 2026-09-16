"""Unified Multi-Model Inference Router for Ollama and OpenRouter."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.config import JarvisConfig, get_config

logger = logging.getLogger("jarvis.core.models")


class ChatMessage(BaseModel):
    """Normalized chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ModelResponse(BaseModel):
    """Standardized response from model inference."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    provider: str  # "ollama" or "openrouter"
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tool_calls: Optional[List[Dict[str, Any]]] = None


class UnifiedModelRouter:
    """Routes prompts and dialogues to Ollama local or OpenRouter cloud."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config or get_config()
        self._client = http_client or httpx.AsyncClient(timeout=60.0)

    async def list_local_models(self) -> List[str]:
        """Query available models from local Ollama instance."""
        try:
            url = f"{self.config.ollama_url.rstrip('/')}/api/tags"
            resp = await self._client.get(url, timeout=5.0)
            if resp.status_code == 200:
                payload = resp.json()
                models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
                return models
        except Exception as exc:
            logger.debug("Ollama list models failed: %s", exc)
        return []

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Execute chat completion using selected provider."""
        start = time.perf_counter()
        chosen_provider = provider or self.config.default_provider
        chosen_model = model or (
            self.config.default_local_model if chosen_provider == "ollama" else self.config.default_cloud_model
        )

        # Auto-detect provider only if not explicitly passed
        if not provider:
            if "/" in chosen_model:
                chosen_provider = "openrouter"
            elif "gemini" in chosen_model.lower():
                chosen_provider = "google"

        if chosen_provider == "ollama":
            try:
                return await self._call_ollama(
                    messages,
                    model=chosen_model,
                    temperature=temperature,
                    start_time=start,
                    tools=tools,
                )
            except Exception as exc:
                logger.warning("Ollama call failed (%s); attempting OpenRouter fallback.", exc)
                if self.config.openrouter_api_key:
                    fallback_model = self.config.default_cloud_model
                    if "/" not in fallback_model:
                        fallback_model = f"google/{fallback_model}"
                    return await self._call_openrouter(
                        messages,
                        model=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        start_time=start,
                        tools=tools,
                    )
                raise RuntimeError(
                    f"Ollama local está offline ou inacessível em {self.config.ollama_url}. "
                    f"Inicie o Ollama no Windows executando 'ollama serve' ou escolha um modelo de nuvem no seletor."
                ) from exc

        if chosen_provider == "google":
            try:
                return await self._call_google(
                    messages,
                    model=chosen_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start,
                )
            except Exception as exc:
                logger.warning("Google Gemini call failed (%s); attempting OpenRouter fallback.", exc)
                if self.config.openrouter_api_key:
                    fallback_model = f"google/{chosen_model}" if "/" not in chosen_model else chosen_model
                    return await self._call_openrouter(
                        messages,
                        model=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        start_time=start,
                        tools=tools,
                    )
                raise

        return await self._call_openrouter(
            messages,
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            start_time=start,
            tools=tools,
        )

    async def _call_ollama(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float,
        start_time: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Call Ollama /api/chat endpoint."""
        url = f"{self.config.ollama_url.rstrip('/')}/api/chat"
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        msg = data.get("message", {})
        text = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ModelResponse(
            text=text,
            model=model,
            provider="ollama",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local inference is free
            tool_calls=tool_calls,
        )

    async def _call_openrouter(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        start_time: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Call OpenRouter chat completions endpoint."""
        api_key = self.config.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://jarvis.local",
            "X-Title": "Jarvis Personal Assistant",
        }
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

        data = resp.json()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        choices = data.get("choices", [])
        text = ""
        tool_calls = None
        if choices:
            message_obj = choices[0].get("message", {})
            text = message_obj.get("content") or ""
            tool_calls = message_obj.get("tool_calls")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Estimate or measure cost
        return ModelResponse(
            text=text,
            model=model,
            provider="openrouter",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
            tool_calls=tool_calls,
        )

    async def _call_google(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        start_time: float,
    ) -> ModelResponse:
        """Call Google Generative Language REST API via GEMINI_API_KEY ($0 marginal free tier)."""
        key = self.config.gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY não configurada no ambiente ou .env.")

        normalized_model = model.replace("google/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent?key={key}"

        system_parts = []
        contents = []
        for m in messages:
            if m.role == "system":
                system_parts.append({"text": m.content})
            elif m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role in ("assistant", "model"):
                contents.append({"role": "model", "parts": [{"text": m.content}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        resp = await self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Google Gemini API error {resp.status_code}: {resp.text}")

        data = resp.json()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return ModelResponse(
            text=text,
            model=normalized_model,
            provider="google",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
        )

