"""Unified Multi-Model Inference Router for Ollama and OpenRouter."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple
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


def normalize_tool_calls(
    raw_calls: Optional[List[Dict[str, Any]]],
    available_tools: Optional[Set[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Normalize tool calls list to standard OpenAI/Jarvis schema."""
    if not raw_calls:
        return None
    normalized: List[Dict[str, Any]] = []
    for i, tcall in enumerate(raw_calls):
        if not isinstance(tcall, dict):
            continue
        fn = tcall.get("function") if isinstance(tcall.get("function"), dict) else tcall
        fn_name = fn.get("name")
        if not fn_name or (available_tools and fn_name not in available_tools):
            continue
        args = fn.get("arguments", {})
        call_id = tcall.get("id") or f"call_{i}_{fn_name}"
        normalized.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
            },
        })
    return normalized if normalized else None


def extract_tool_calls_from_text(
    text: str,
    available_tools: Optional[Set[str]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Extract function/tool calls embedded in text from models (e.g. Qwen/Ollama).
    Returns (tool_calls, cleaned_text).
    """
    if not text or not text.strip():
        return None, text

    extracted: List[Dict[str, Any]] = []
    cleaned_text = text

    # 1. Look for <tool_call>...</tool_call> tags
    tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if tool_call_blocks:
        for block in tool_call_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict) and "name" in parsed:
                    extracted.append(parsed)
                elif isinstance(parsed, list):
                    extracted.extend([item for item in parsed if isinstance(item, dict) and "name" in item])
            except Exception:
                pass
        cleaned_text = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned_text, flags=re.DOTALL).strip()

    # 2. Look for markdown json code blocks: ```json ... ```
    if not extracted:
        code_blocks = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text))
        for match in code_blocks:
            block = match.group(1).strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "name" in parsed:
                    extracted.append(parsed)
                elif isinstance(parsed, list):
                    extracted.extend([item for item in parsed if isinstance(item, dict) and "name" in item])
            except Exception:
                pass
        if extracted:
            cleaned_text = re.sub(r"```(?:json)?\s*[\s\S]*?\s*```", "", cleaned_text).strip()

    # 3. Look for JSON array: [{"name": ...}]
    if not extracted:
        array_match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            extracted.append(item)
                    cleaned_text = text[:array_match.start()] + text[array_match.end():]
                    cleaned_text = cleaned_text.strip()
            except Exception:
                pass

    # 4. Stream / sequence of JSON objects: {"name": ..., "arguments": ...}
    if not extracted:
        decoder = json.JSONDecoder()
        idx = 0
        search_text = text
        spans_to_remove = []
        while idx < len(search_text):
            match = re.search(r"\{\s*\"name\"\s*:", search_text[idx:])
            if not match:
                break
            start_pos = idx + match.start()
            try:
                obj, end_idx = decoder.raw_decode(search_text[start_pos:])
                if isinstance(obj, dict) and "name" in obj:
                    extracted.append(obj)
                    spans_to_remove.append((start_pos, start_pos + end_idx))
                idx = start_pos + end_idx
            except Exception:
                idx = start_pos + 1

        if spans_to_remove:
            parts = []
            last_end = 0
            for start, end in spans_to_remove:
                parts.append(search_text[last_end:start])
                last_end = end
            parts.append(search_text[last_end:])
            cleaned_text = "".join(parts).strip()

    norm = normalize_tool_calls(extracted, available_tools=available_tools)
    if norm:
        return norm, cleaned_text
    return None, text


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
        allow_cloud_fallback: bool = True,
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
                if not allow_cloud_fallback:
                    logger.warning("Ollama call failed (%s) and cloud fallback is blocked by circuit breaker policy. Retrying local Ollama once...", exc)
                    import asyncio
                    await asyncio.sleep(1.0)
                    try:
                        return await self._call_ollama(
                            messages,
                            model=chosen_model,
                            temperature=temperature,
                            start_time=start,
                            tools=tools,
                        )
                    except Exception as retry_exc:
                        raise RuntimeError(
                            f"Ollama local está ocupado ou inacessível ({retry_exc}). "
                            f"O fallback para APIs pagas em nuvem foi bloqueado porque o circuit breaker de orçamento está ativo."
                        ) from retry_exc

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

        available_tool_names: Optional[Set[str]] = None
        if tools:
            available_tool_names = set()
            for t in tools:
                if isinstance(t, dict):
                    fn_name = t.get("function", {}).get("name") or t.get("name")
                    if fn_name:
                        available_tool_names.add(fn_name)

        if tool_calls:
            tool_calls = normalize_tool_calls(tool_calls, available_tools=available_tool_names)
        elif tools and text:
            extracted_calls, remaining_text = extract_tool_calls_from_text(
                text, available_tools=available_tool_names
            )
            if extracted_calls:
                tool_calls = extracted_calls
                text = remaining_text

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

        available_tool_names = None
        if tools:
            available_tool_names = set()
            for t in tools:
                if isinstance(t, dict):
                    fn_name = t.get("function", {}).get("name") or t.get("name")
                    if fn_name:
                        available_tool_names.add(fn_name)

        if tool_calls:
            tool_calls = normalize_tool_calls(tool_calls, available_tools=available_tool_names)
        elif tools and text:
            extracted_calls, remaining_text = extract_tool_calls_from_text(
                text, available_tools=available_tool_names
            )
            if extracted_calls:
                tool_calls = extracted_calls
                text = remaining_text

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost_usd = self._estimate_cost(model, prompt_tokens, completion_tokens)
        return ModelResponse(
            text=text,
            model=model,
            provider="openrouter",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            tool_calls=tool_calls,
        )

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate inference cost in USD based on model pricing."""
        lowered = model.lower()
        if "ollama" in lowered or "local" in lowered:
            return 0.0
        try:
            from jarvis.core.telemetry import PRICING_PER_1M
            rates = (0.50, 1.50)
            for key, val in PRICING_PER_1M.items():
                if key in lowered:
                    rates = val
                    break
            prompt_cost = (prompt_tokens / 1_000_000) * rates[0]
            comp_cost = (completion_tokens / 1_000_000) * rates[1]
            return round(prompt_cost + comp_cost, 6)
        except Exception:
            return 0.0

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
        cost_usd = 0.0  # Direct Google Gemini Developer API key is $0 marginal on free tier

        return ModelResponse(
            text=text,
            model=normalized_model,
            provider="google",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

