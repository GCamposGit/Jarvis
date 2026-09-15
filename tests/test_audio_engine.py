"""Deterministic tests for Jarvis Audio Transcription Engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock
import httpx
import pytest

from jarvis.core.audio import AudioTranscriptionEngine, TranscriptionResult
from jarvis.core.config import JarvisConfig


class MockWhisperSegment:
    def __init__(self, text: str, avg_logprob: float) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class MockWhisperInfo:
    def __init__(self, duration: float = 2.5, language: str = "pt") -> None:
        self.duration = duration
        self.language = language


class MockWhisperModelGood:
    def transcribe(self, *args, **kwargs):
        segments = [
            MockWhisperSegment("Olá Jarvis, tudo bem?", avg_logprob=-0.2),
            MockWhisperSegment("Inicie uma demanda para a Dark Factory.", avg_logprob=-0.15),
        ]
        return segments, MockWhisperInfo(duration=3.2)


class MockWhisperModelPoorQuality:
    def transcribe(self, *args, **kwargs):
        # Very low confidence (avg_logprob < -1.0)
        segments = [
            MockWhisperSegment("...ruído estático...", avg_logprob=-1.85),
        ]
        return segments, MockWhisperInfo(duration=1.0)


def test_audio_file_not_found():
    engine = AudioTranscriptionEngine()
    with pytest.raises(FileNotFoundError):
        engine.transcribe("non_existent_audio_file.wav")


def test_local_transcription_high_confidence(tmp_path: Path):
    dummy_wav = tmp_path / "sample.wav"
    dummy_wav.write_bytes(b"RIFFdummywavecontent")

    cfg = JarvisConfig(whisper_confidence_threshold=-1.0)
    engine = AudioTranscriptionEngine(config=cfg, local_model_instance=MockWhisperModelGood())

    result = engine.transcribe(dummy_wav)
    assert isinstance(result, TranscriptionResult)
    assert result.engine_used == "local_faster_whisper"
    assert not result.fallback_triggered
    assert "Olá Jarvis" in result.text
    assert result.avg_logprob > -0.5
    assert result.confidence_score > 0.8


def test_fallback_triggered_on_poor_quality(tmp_path: Path):
    dummy_wav = tmp_path / "noisy.wav"
    dummy_wav.write_bytes(b"RIFFdummywavecontent")

    # Mock Groq fallback response
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={"text": "Texto transcrito com alta qualidade pelo Groq Whisper."},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))

    cfg = JarvisConfig(
        groq_api_key="gsk_test_mock_key_12345",
        whisper_confidence_threshold=-1.0,
    )
    engine = AudioTranscriptionEngine(
        config=cfg,
        local_model_instance=MockWhisperModelPoorQuality(),
        http_client=mock_client,
    )

    result = engine.transcribe(dummy_wav)
    assert result.fallback_triggered
    assert "Groq" in result.text
    assert result.engine_used == "cloud_groq_whisper"
    assert "confidence score too low" in (result.fallback_reason or "").lower()


def test_fallback_triggered_when_local_whisper_fails(tmp_path: Path):
    dummy_wav = tmp_path / "sample.wav"
    dummy_wav.write_bytes(b"RIFFdummywavecontent")

    class CrashingModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("CUDA OOM memory error")

    def mock_handler(request: httpx.Request):
        return httpx.Response(200, json={"text": "Recuperado via Groq fallback"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    cfg = JarvisConfig(groq_api_key="gsk_key")
    engine = AudioTranscriptionEngine(
        config=cfg,
        local_model_instance=CrashingModel(),
        http_client=mock_client,
    )

    result = engine.transcribe(dummy_wav)
    assert result.fallback_triggered
    assert result.engine_used == "cloud_groq_whisper"
    assert "Recuperado" in result.text


def test_bilingual_groq_fallback_pt_and_en(tmp_path: Path):
    dummy_wav = tmp_path / "bilingual.wav"
    dummy_wav.write_bytes(b"RIFFdummywavecontent")

    captured_requests = []

    def mock_handler(request: httpx.Request):
        captured_requests.append(request)
        # Check if english or portuguese
        return httpx.Response(200, json={"text": "Voice transcribed successfully"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    cfg = JarvisConfig(groq_api_key="gsk_test_mock_123")
    engine = AudioTranscriptionEngine(
        config=cfg,
        local_model_instance=MockWhisperModelPoorQuality(),
        http_client=mock_client,
    )

    # Test English
    res_en = engine.transcribe(dummy_wav, language="en")
    assert res_en.fallback_triggered
    assert res_en.language == "en"
    assert res_en.engine_used == "cloud_groq_whisper"
    assert res_en.text == "Voice transcribed successfully"

    # Test Portuguese
    res_pt = engine.transcribe(dummy_wav, language="pt")
    assert res_pt.fallback_triggered
    assert res_pt.language == "pt"
    assert res_pt.engine_used == "cloud_groq_whisper"
    assert len(captured_requests) == 2
