"""Shared auth for the local endpoints.

The llama-swap router listens on 0.0.0.0 so it works over LAN and Tailscale,
which means it needs a key — see configs/secrets/auth.yaml (gitignored).

Resolution order, first hit wins:
  1. $LAB_API_KEY          — override for one-off runs
  2. configs/secrets/api-key.txt
  3. nothing               — headers come back without Authorization, which is
                             correct for a router started without apiKeys and
                             for SGLang when it was launched without --api-key.

Returning no header rather than a bogus one matters: llama-swap rejects a wrong
key with 401 but allows a missing one when no apiKeys are configured, so an
empty value would break the unauthenticated case that still needs to work.
"""

from __future__ import annotations

import os
from pathlib import Path

_KEY_FILE = Path(__file__).resolve().parent.parent / "configs" / "secrets" / "api-key.txt"


def api_key() -> str:
    env = os.environ.get("LAB_API_KEY", "").strip()
    if env:
        return env
    try:
        return _KEY_FILE.read_text().strip()
    except OSError:
        return ""


def headers(extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h
