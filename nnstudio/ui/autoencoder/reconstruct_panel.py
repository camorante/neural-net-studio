"""Autoencoder stage 4: look at what came back out.

Three questions, three buttons. What does a reconstruction look like? Which
images does the model rebuild worst? And does the class it never saw rebuild
worse than the ones it did - which is anomaly detection with no labels.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import hint, scrollable


class AutoencoderReconstructPanel(QWidget):
    """Sampling controls plus the readout."""

    reconstruct_requested = pyqtSignal(int)
    hardest_requested = pyqtSignal(int)
    anomaly_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_controls_group())
        inner.addWidget(self._build_result_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

    # ------------------------------------------------------------------- build

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Look at the output")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Images to show:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(4, 16)
        self.count_spin.setValue(8)
        row.addWidget(self.count_spin, 1)
        layout.addLayout(row)

        self.reconstruct_button = QPushButton("Show reconstructions")
        self.reconstruct_button.setObjectName("Primary")
        self.reconstruct_button.setMinimumHeight(32)
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.clicked.connect(
            lambda: self.reconstruct_requested.emit(self.count_spin.value())
        )
        layout.addWidget(self.reconstruct_button)

        self.hardest_button = QPushButton("Show the hardest to rebuild")
        self.hardest_button.setMinimumHeight(30)
        self.hardest_button.setEnabled(False)
        self.hardest_button.clicked.connect(
            lambda: self.hardest_requested.emit(self.count_spin.value())
        )
        self.hardest_button.setToolTip(
            "Ranks every validation image by reconstruction error and shows the "
            "worst. These are the images the model found least familiar."
        )
        layout.addWidget(self.hardest_button)

        self.anomaly_button = QPushButton("Score the held-out class")
        self.anomaly_button.setMinimumHeight(30)
        self.anomaly_button.setEnabled(False)
        self.anomaly_button.clicked.connect(self.anomaly_requested.emit)
        self.anomaly_button.setToolTip(
            "Only available when a class was held out of training in the "
            "Architecture tab."
        )
        layout.addWidget(self.anomaly_button)

        layout.addWidget(
            hint(
                "The number under each image is its own reconstruction error. "
                "Read a column top to bottom: same picture, different waist."
            )
        )
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Result")
        layout = QVBoxLayout(group)

        self.result_label = QLabel("Train an autoencoder first.")
        self.result_label.setObjectName("Result")
        self.result_label.setWordWrap(True)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.result_label)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Hint")
        self.detail_label.setWordWrap(True)
        self.detail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.detail_label)
        return group

    # ------------------------------------------------------------------ public

    def set_ready(self, ready: bool) -> None:
        self.reconstruct_button.setEnabled(ready)
        self.hardest_button.setEnabled(ready)
        if not ready:
            self.anomaly_button.setEnabled(False)
            self.result_label.setText("Train an autoencoder first.")
            self.detail_label.setText("")

    def set_anomaly_available(self, available: bool) -> None:
        self.anomaly_button.setEnabled(available)

    def show_result(self, headline: str, detail: str = "") -> None:
        self.result_label.setText(headline)
        self.detail_label.setText(detail)
