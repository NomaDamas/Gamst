"""Optional realtime voice consumer configuration.

This module deliberately does not open a microphone, create a websocket, or
implement full duplex. GAMST first extracts local structured audio Context.
An application may enable a commercial realtime voice consumer explicitly when
it wants a provider to render/respond from that Context.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    """Opt-in provider metadata; no network action occurs at construction."""

    enabled: bool = False
    provider: str = "none"
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    context_only: bool = True

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.context_only) is not bool:
            raise ValueError("enabled/context_only must be boolean")
        if not isinstance(self.provider, str) or self.provider not in ("none", "openai", "gemini", "custom"):
            raise ValueError("provider must be none, openai, gemini, or custom")
        if not self.enabled:
            if self.provider != "none":
                raise ValueError("disabled realtime voice must use provider='none'")
            return
        if self.provider == "none":
            raise ValueError("enabled realtime voice requires a provider")
        if not isinstance(self.endpoint, str) or urlparse(self.endpoint).scheme not in ("http", "https"):
            raise ValueError("enabled realtime voice requires an http(s) endpoint")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("enabled realtime voice requires a model")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError("enabled realtime voice requires api_key_env")

    def status(self) -> dict:
        return {"enabled": self.enabled, "provider": self.provider,
                "model": self.model, "context_only": self.context_only,
                "network_started": False}
