"""Sequence stage 4: look at what the model actually answered.

The forecast view is the one with a trap in it. A prediction drawn alone against
the truth looks superb even when the model has learned nothing, because a curve
that repeats the previous value tracks the target almost perfectly by eye. So
persistence is drawn on the same axes, always, and the button says so.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import hint, scrollable


class SequencePredictPanel(QWidget):
    """Buttons that show the model's answers on the validation set."""

    forecast_requested = pyqtSignal(int)
    confusion_requested = pyqtSignal()
    windows_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_views_group())
        inner.addWidget(self._build_result_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self.set_ready(False)

    # ------------------------------------------------------------------- build

    def _build_views_group(self) -> QGroupBox:
        group = QGroupBox("Views")
        grid = QGridLayout(group)

        self.windows_button = QPushButton("Show a few raw windows")
        self.windows_button.setMinimumHeight(30)
        self.windows_button.clicked.connect(self.windows_requested.emit)
        grid.addWidget(self.windows_button, 0, 0, 1, 2)

        grid.addWidget(QLabel("Steps to plot:"), 1, 0)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(20, 600)
        self.count_spin.setSingleStep(20)
        self.count_spin.setValue(120)
        grid.addWidget(self.count_spin, 1, 1)

        self.forecast_button = QPushButton("Forecast vs persistence")
        self.forecast_button.setObjectName("Primary")
        self.forecast_button.setMinimumHeight(32)
        self.forecast_button.clicked.connect(
            lambda: self.forecast_requested.emit(int(self.count_spin.value()))
        )
        grid.addWidget(self.forecast_button, 2, 0, 1, 2)

        self.confusion_button = QPushButton("Confusion matrix")
        self.confusion_button.setMinimumHeight(30)
        self.confusion_button.clicked.connect(self.confusion_requested.emit)
        grid.addWidget(self.confusion_button, 3, 0, 1, 2)

        grid.addWidget(
            hint(
                "The forecast chart always draws persistence alongside the "
                "model. A prediction that merely lags the truth by one step "
                "looks like a near-perfect fit on its own, and that is the "
                "single most common way a forecasting result flatters itself."
            ),
            4, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Result")
        layout = QVBoxLayout(group)
        self.result_label = QLabel("Train a model first.")
        self.result_label.setObjectName("Metric")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Hint")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        return group

    # ------------------------------------------------------------------ public

    def set_ready(self, ready: bool, task_kind: str = "") -> None:
        """Only the view that suits the current task is offered."""
        self.forecast_button.setEnabled(ready and task_kind == "forecast")
        self.confusion_button.setEnabled(ready and task_kind == "classify")
        self.count_spin.setEnabled(ready and task_kind == "forecast")
        if not ready:
            self.result_label.setText("Train a model first.")
            self.detail_label.setText("")

    def set_has_data(self, has_data: bool) -> None:
        self.windows_button.setEnabled(bool(has_data))

    def show_result(self, headline: str, detail: str = "") -> None:
        self.result_label.setText(headline)
        self.detail_label.setText(detail)
