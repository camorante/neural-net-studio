"""CNN stage 4: run the trained model over validation images and read it."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from ..widgets import hint, scrollable


class VisionPredictPanel(QWidget):
    """Sample validation images and show what the network says about them."""

    predict_requested = pyqtSignal(int)

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

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Sample")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Images to show:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(4, 24)
        self.count_spin.setSingleStep(2)
        self.count_spin.setValue(12)
        row.addWidget(self.count_spin, 1)
        layout.addLayout(row)

        self.predict_button = QPushButton("Show predictions")
        self.predict_button.setObjectName("Primary")
        self.predict_button.setMinimumHeight(32)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(
            lambda: self.predict_requested.emit(self.count_spin.value())
        )
        layout.addWidget(self.predict_button)

        layout.addWidget(
            hint(
                "Draws a fresh random batch from the validation split each time. "
                "A green frame means the prediction matched, red means it did not."
            )
        )
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Result")
        layout = QVBoxLayout(group)
        self.result_label = QLabel("Train the network first.")
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
        self.predict_button.setEnabled(ready)
        if not ready:
            self.result_label.setText("Train the network first.")
            self.detail_label.setText("")

    def show_result(self, headline: str, detail: str = "") -> None:
        self.result_label.setText(headline)
        self.detail_label.setText(detail)
