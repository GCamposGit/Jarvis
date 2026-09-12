"""Direct CLI Runner for Jarvis Assistant Web Server."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure jarvis package root is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
from jarvis.core.config import get_config
from jarvis.web.app import create_app


def main() -> None:
    config = get_config()
    print("=" * 60)
    print("  JARVIS PERSONAL PRODUCTIVITY ASSISTANT (v0.1.0)")
    print("=" * 60)
    print(f"  URL Local:   http://{config.host}:{config.port}")
    print(f"  DarkHub:     {config.darkhub_url}")
    print(f"  Ollama:      {config.ollama_url}")
    print(f"  Áudio Local: faster-whisper ({config.whisper_model_size})")
    print(f"  Fallback:    Groq / OpenRouter")
    print("=" * 60)
    print("Iniciando servidor FastAPI / Uvicorn... Pressione Ctrl+C para encerrar.\n")

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
