"""Shared setup for every suite: import path, quiet TensorFlow, output folder.

Each suite is a standalone script rather than a pytest module, on purpose. They
train real networks and drive real widgets, so they read as transcripts: run one
and the printed numbers are the evidence. This module is the only thing they
share, and it exists so that no suite has to hardcode a path to the checkout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "_output"

# Keras prints a banner and a stream of oneDNN notices that bury the assertions.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WINDOWS_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


def offscreen() -> None:
    """Render Qt into memory instead of onto a screen.

    Must run before PyQt6 is imported: the platform plugin is chosen once, at
    import time, and cannot be swapped afterwards.

    The offscreen plugin ships with no font database of its own, so without
    QT_QPA_FONTDIR it falls back to a font whose widths are roughly 1.8x the
    real ones - measured on the transformer's training panel as 544px against
    301px. Pointing it at the system fonts makes widths trustworthy and the
    screenshots readable. setdefault, so a caller can still override it.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if WINDOWS_FONTS.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(WINDOWS_FONTS))


def output_dir() -> Path:
    """Where screenshots and generated fixtures go. Never committed."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT
