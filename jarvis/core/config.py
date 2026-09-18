"""Configuration and Environment Settings for Jarvis using pure Pydantic v2."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


def _recover_env_or_registry(key: str) -> Optional[str]:
    """Recover environment variable from process env, .env file or Windows user registry."""
    val = os.environ.get(key)
    if not val:
        candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
        for env_file in candidates:
            if env_file.is_file():
                try:
                    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == key:
                                val = v.strip().strip("'\"")
                                break
                except Exception:
                    pass
            if val:
                break

    if not val and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                val, _ = winreg.QueryValueEx(k, key)
        except Exception:
            pass
    return val.strip() if (val and val.strip()) else None


class MCPServerConfig(BaseModel):
    """Configuration for an individual MCP server connection."""

    name: str
    transport: str = "stdio"  # "stdio" or "sse"
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None  # for SSE transport
    enabled: bool = True


class JarvisConfig(BaseModel):
    """Global configuration for Jarvis assistant."""

    model_config = ConfigDict(extra="ignore")

    # Server settings
    host: str = Field(default_factory=lambda: os.environ.get("JARVIS_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.environ.get("JARVIS_PORT", "8765")))
    debug: bool = Field(default_factory=lambda: os.environ.get("JARVIS_DEBUG", "").lower() in ("1", "true"))

    # External Integration URLs
    darkhub_url: str = Field(
        default_factory=lambda: os.environ.get("DARKHUB_URL", "https://darkhub.ggcampos.com")
    )
    ollama_url: str = Field(
        default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434")
    )

    # API Keys
    openrouter_api_key: Optional[str] = Field(
        default_factory=lambda: _recover_env_or_registry("OPENROUTER_API_KEY")
    )
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: _recover_env_or_registry("GROQ_API_KEY")
    )
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: _recover_env_or_registry("GEMINI_API_KEY")
    )

    # Model defaults
    default_provider: str = Field(default_factory=lambda: os.environ.get("JARVIS_DEFAULT_PROVIDER", "ollama"))
    default_local_model: str = Field(default_factory=lambda: os.environ.get("JARVIS_DEFAULT_LOCAL_MODEL", "qwen-code-deep:latest"))
    default_cloud_model: str = Field(default_factory=lambda: os.environ.get("JARVIS_DEFAULT_CLOUD_MODEL", "gemini-2.5-flash"))

    # Audio Engine settings
    whisper_model_size: str = Field(default_factory=lambda: os.environ.get("JARVIS_WHISPER_MODEL", "base"))
    whisper_device: str = Field(default_factory=lambda: os.environ.get("JARVIS_WHISPER_DEVICE", "auto"))
    whisper_confidence_threshold: float = Field(
        default_factory=lambda: float(os.environ.get("JARVIS_WHISPER_THRESHOLD", "-1.0"))
    )
    audio_upload_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".jarvis" / "audio"
    )

    # Memory and Second Brain Settings
    memory_db_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("JARVIS_MEMORY_DB", str(Path.cwd() / ".jarvis" / "memory.db")))
    )

    # MCP Servers for Second Brain
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)


def get_config() -> JarvisConfig:
    """Factory creating the current Jarvis configuration."""
    cfg = JarvisConfig()
    cfg.audio_upload_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg
