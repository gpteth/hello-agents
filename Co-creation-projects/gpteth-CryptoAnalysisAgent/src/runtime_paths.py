"""Writable paths. Vercel functions can only write under /tmp."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/crypto-analysis-agent")


def on_vercel() -> bool:
    """True on Vercel. The Python runtime may omit VERCEL, and os.access can
    still report the read-only bundle as writable."""
    if os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return True
    root = str(PROJECT_ROOT)
    if root.startswith("/var/task") or root.startswith("/var/lang"):
        return True
    try:
        probe = PROJECT_ROOT / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return True
    return False


def reports_dir() -> Path:
    override = (os.getenv("OUTPUT_DIR") or "").strip()
    if override:
        return Path(override)
    if on_vercel():
        return TMP_ROOT / "outputs" / "reports"
    return PROJECT_ROOT / "outputs" / "reports"


def signals_path() -> str:
    override = (os.getenv("SIGNALS_PATH") or "").strip()
    if override:
        return override
    if on_vercel():
        return str(TMP_ROOT / "outputs" / "signals.jsonl")
    return "outputs/signals.jsonl"


def workspace_dir() -> str:
    override = (os.getenv("WORKSPACE_PATH") or "").strip()
    if on_vercel():
        # A relative path from .env points at the read-only function bundle.
        if override.startswith("/tmp/"):
            return override
        return str(TMP_ROOT / "workspace")
    if override:
        return override
    return "./workspace"
