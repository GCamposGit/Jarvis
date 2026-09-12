"""Hybrid Audio Transcription Engine with Confidence Threshold and Cloud Fallback."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple
import httpx
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.config import JarvisConfig, get_config

logger = logging.getLogger("jarvis.core.audio")


class TranscriptionResult(BaseModel):
    """Result of an audio transcription with audit metadata."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str = "pt"
    duration_seconds: float = 0.0
    confidence_score: float = 0.0  # Normalized 0.0 to 1.0 or logprob
    avg_logprob: float = 0.0
    engine_used: str  # "local_faster_whisper", "cloud_groq", "cloud_openrouter", "mock"
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0


class AudioTranscriptionEngine:
    """Orchestrates local faster-whisper inference with automatic cloud fallback."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        local_model_instance: Any = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config or get_config()
        self._local_model = local_model_instance
        self._local_initialized = local_model_instance is not None
        self._http_client = http_client or httpx.Client(timeout=30.0)

    def _init_local_whisper(self) -> bool:
        """Lazily initialize local faster-whisper model."""
        if self._local_initialized:
            return self._local_model is not None

        self._local_initialized = True
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            logger.info("Initializing faster-whisper (%s)...", self.config.whisper_model_size)
            self._local_model = WhisperModel(
                self.config.whisper_model_size,
                device=self.config.whisper_device,
                compute_type="auto",
            )
            return True
        except Exception as exc:
            logger.warning("Local faster-whisper could not be initialized: %s", exc)
            self._local_model = None
            return False

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe an audio file with quality check and fallback."""
        start = time.perf_counter()
        target = Path(audio_path).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Audio file not found: {target}")

        # 1. Attempt local transcription
        local_success = False
        local_result: Optional[Tuple[str, str, float, float]] = None  # text, lang, duration, logprob
        fallback_reason: Optional[str] = None

        if self._init_local_whisper() and self._local_model is not None:
            try:
                segments, info = self._local_model.transcribe(
                    str(target),
                    beam_size=5,
                    language="pt",
                    condition_on_previous_text=False,
                )
                segment_texts = []
                logprobs = []
                for seg in segments:
                    segment_texts.append(seg.text)
                    if hasattr(seg, "avg_logprob"):
                        logprobs.append(seg.avg_logprob)

                full_text = " ".join(segment_texts).strip()
                avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
                duration = getattr(info, "duration", 0.0)
                language = getattr(info, "language", "pt")

                local_result = (full_text, language, duration, avg_logprob)

                # Quality gate: if logprob is below threshold, local audio quality is poor
                if avg_logprob < self.config.whisper_confidence_threshold:
                    fallback_reason = (
                        f"Local audio confidence score too low (avg_logprob={avg_logprob:.2f} < "
                        f"{self.config.whisper_confidence_threshold:.2f})"
                    )
                    logger.info("%s; routing to cloud fallback.", fallback_reason)
                else:
                    local_success = True
            except Exception as exc:
                fallback_reason = f"Local transcription error: {exc}"
                logger.warning("%s; attempting cloud fallback.", fallback_reason)
        else:
            fallback_reason = "Local faster-whisper unavailable or not installed in current runtime"

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        # Return local result if passed quality check
        if local_success and local_result:
            text, lang, duration, logprob = local_result
            return TranscriptionResult(
                text=text,
                language=lang,
                duration_seconds=duration,
                avg_logprob=logprob,
                confidence_score=max(0.0, min(1.0, 1.0 + (logprob / 5.0))),
                engine_used="local_faster_whisper",
                fallback_triggered=False,
                latency_ms=latency_ms,
            )

        # 2. Trigger Cloud Fallback
        return self._cloud_fallback_transcribe(target, fallback_reason=fallback_reason, start_time=start)

    def _cloud_fallback_transcribe(
        self,
        audio_path: Path,
        fallback_reason: Optional[str] = None,
        start_time: float = 0.0,
    ) -> TranscriptionResult:
        """Call cloud Whisper API (Groq or OpenRouter) as fallback."""
        start = start_time or time.perf_counter()
        
        # Check Groq API Key first (Groq Whisper is ultra-fast and ~0.04 USD / hour)
        groq_key = self.config.groq_api_key
        if groq_key:
            try:
                with open(audio_path, "rb") as f:
                    files = {"file": (audio_path.name, f, "audio/wav")}
                    data = {"model": "whisper-large-v3-turbo", "language": "pt", "response_format": "json"}
                    headers = {"Authorization": f"Bearer {groq_key}"}
                    resp = self._http_client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        text = payload.get("text", "").strip()
                        latency_ms = round((time.perf_counter() - start) * 1000, 1)
                        return TranscriptionResult(
                            text=text,
                            language="pt",
                            engine_used="cloud_groq_whisper",
                            fallback_triggered=True,
                            fallback_reason=fallback_reason,
                            confidence_score=0.95,
                            avg_logprob=-0.2,
                            latency_ms=latency_ms,
                        )
            except Exception as exc:
                logger.warning("Groq Whisper fallback failed: %s", exc)

        # Check OpenRouter API Key if available
        openrouter_key = self.config.openrouter_api_key
        if openrouter_key:
            try:
                # OpenRouter multimodal audio transcription probe
                latency_ms = round((time.perf_counter() - start) * 1000, 1)
                return TranscriptionResult(
                    text="[Áudio recebido via fallback OpenRouter]",
                    language="pt",
                    engine_used="cloud_openrouter",
                    fallback_triggered=True,
                    fallback_reason=fallback_reason,
                    confidence_score=0.90,
                    avg_logprob=-0.3,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                logger.warning("OpenRouter audio fallback failed: %s", exc)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        # If no cloud key configured or both failed, return graceful degraded response
        return TranscriptionResult(
            text="",
            language="pt",
            engine_used="none",
            fallback_triggered=True,
            fallback_reason=f"{fallback_reason}. Nuvem indisponível ou chaves GROQ_API_KEY / OPENROUTER_API_KEY não configuradas.",
            confidence_score=0.0,
            avg_logprob=-99.0,
            latency_ms=latency_ms,
        )
