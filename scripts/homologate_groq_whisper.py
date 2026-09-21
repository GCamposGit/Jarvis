"""Homologation probe for the Groq Cloud Whisper fallback (Dark Factory ticket JRV-01).

Validates the cloud contingency route end-to-end against the real Groq API:
generates a short synthetic WAV (no external audio fixture required), forces
the transcription engine through AudioTranscriptionEngine and asserts the
call landed on `cloud_groq_whisper`.

Usage:
    python scripts/homologate_groq_whisper.py

Requires GROQ_API_KEY to be set in `.env`, the process environment, or the
Windows user registry (see jarvis.core.config._recover_env_or_registry).
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.core.audio import AudioTranscriptionEngine
from jarvis.core.config import get_config


def _write_probe_wav(path: Path, duration_s: float = 1.5, freq_hz: float = 440.0) -> None:
    """Write a short mono 16kHz sine-tone WAV so the API has real audio bytes to accept."""
    sample_rate = 16000
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(3000 * math.sin(2 * math.pi * freq_hz * (i / sample_rate)))
            frames.extend(struct.pack("<h", value))
        wav_file.writeframes(bytes(frames))


def main() -> int:
    config = get_config()
    if not config.groq_api_key:
        print(
            "[FALHA] GROQ_API_KEY não configurada. Defina-a no .env, na variável de "
            "ambiente ou no registro do Windows antes de rodar esta homologação.",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        probe_path = Path(tmp_dir) / "jrv01_probe.wav"
        _write_probe_wav(probe_path)

        engine = AudioTranscriptionEngine(config=config)
        result = engine._cloud_fallback_transcribe(
            probe_path,
            language="pt",
            fallback_reason="JRV-01 homologation probe (forced cloud path)",
        )

    print(f"engine_used      = {result.engine_used}")
    print(f"fallback_reason  = {result.fallback_reason}")
    print(f"latency_ms       = {result.latency_ms}")
    print(f"text             = {result.text!r}")

    if result.engine_used == "cloud_groq_whisper":
        print("\n[SUCESSO] Rota de contingência Groq Whisper (whisper-large-v3-turbo) homologada.")
        return 0

    print(
        "\n[FALHA] A chamada não retornou pela rota cloud_groq_whisper. "
        "Verifique a validade da GROQ_API_KEY e a disponibilidade de api.groq.com.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
