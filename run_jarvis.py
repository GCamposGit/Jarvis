from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Ensure jarvis package root is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
from jarvis.core.config import get_config
from jarvis.web.app import create_app


def _is_server_healthy(url: str, timeout: float = 1.0) -> bool:
    """Check if Jarvis server is already responding."""
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _launch_browser_when_ready(url: str, max_wait_seconds: float = 15.0) -> None:
    """Polls local server in a background thread and opens the default browser as soon as ready."""
    def _worker() -> None:
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            if _is_server_healthy(url, timeout=0.5):
                webbrowser.open(url)
                return
            time.sleep(0.3)
        # Fallback: open if timeout reached
        webbrowser.open(url)

    thread = threading.Thread(target=_worker, name="jarvis-browser-launcher", daemon=True)
    thread.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis Assistant Runner")
    parser.add_argument(
        "--open",
        "-o",
        "--open-browser",
        dest="open_browser",
        action="store_true",
        default=False,
        help="Abrir o navegador automaticamente assim que o servidor estiver pronto.",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            os.system("title Jarvis Assistant v0.1.0")
        except Exception:
            pass

    config = get_config()
    server_url = f"http://{config.host}:{config.port}"

    # If the server is already running, avoid port conflict and simply open browser
    if _is_server_healthy(server_url):
        print(f"\n[INFO] Jarvis ja esta em execucao em {server_url}.")
        if args.open_browser:
            print("[INFO] Abrindo navegador...")
            webbrowser.open(server_url)
        return

    print("=" * 60)
    print("  JARVIS PERSONAL PRODUCTIVITY ASSISTANT (v0.1.0)")
    print("=" * 60)
    print(f"  URL Local:   {server_url}")
    print(f"  DarkHub:     {config.darkhub_url}")
    print(f"  Ollama:      {config.ollama_url}")
    print(f"  Audio Local: faster-whisper ({config.whisper_model_size})")
    print(f"  Fallback:    Groq / OpenRouter")
    print("=" * 60)

    if args.open_browser:
        print("  Auto-Open:   Navegador sera aberto automaticamente assim que pronto.")
        _launch_browser_when_ready(server_url)

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
