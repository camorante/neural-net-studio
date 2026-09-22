"""Neural Net Studio - build, train and inspect Keras networks interactively."""
from __future__ import annotations

import os
import sys

# Quiet TensorFlow before anything can import it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from PyQt6.QtWidgets import QApplication

from nnstudio.ui.main_window import MainWindow
from nnstudio.ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Neural Net Studio")
    app.setOrganizationName("nnstudio")
    apply_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
