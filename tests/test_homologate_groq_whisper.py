"""Tests for the JRV-01 Groq Whisper homologation probe script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import httpx
import pytest

from jarvis.core.config import JarvisConfig

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "homologate_groq_whisper.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("homologate_groq_whisper", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


def test_main_fails_fast_without_groq_api_key(script_module, capsys):
    with mock.patch.object(script_module, "get_config", return_value=JarvisConfig(groq_api_key=None)):
        exit_code = script_module.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "GROQ_API_KEY não configurada" in captured.err


def test_main_succeeds_with_mocked_groq_response(script_module, capsys):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/audio/transcriptions"
        return httpx.Response(200, json={"text": "homologação ok"})

    cfg = JarvisConfig(groq_api_key="gsk_test_mock_key")

    with mock.patch.object(script_module, "get_config", return_value=cfg), mock.patch.object(
        script_module.AudioTranscriptionEngine,
        "__init__",
        lambda self, config=None: setattr(self, "config", cfg)
        or setattr(self, "_http_client", httpx.Client(transport=httpx.MockTransport(mock_handler))),
    ):
        exit_code = script_module.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SUCESSO" in captured.out
    assert "cloud_groq_whisper" in captured.out


def test_write_probe_wav_produces_valid_wave_file(script_module, tmp_path):
    import wave

    probe_path = tmp_path / "probe.wav"
    script_module._write_probe_wav(probe_path, duration_s=0.2)

    assert probe_path.is_file()
    with wave.open(str(probe_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() > 0
