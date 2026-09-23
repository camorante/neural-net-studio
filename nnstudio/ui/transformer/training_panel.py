"""Transformer stage 3: run controls, live metrics and the log.

Every run ends with the attention map being tested, not just drawn, so the log
of an ordinary Train already carries the faithfulness verdict. The second button
is the position probe - the transformer's version of asking whether order
mattered, answered by removing the one thing that lets attention see it.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.transformer_trainer import DEFAULT_EPOCHS
from ..widgets import hint, scrollable


class TransformerTrainingPanel(QWidget):
    """Run controls for the transformer workspace."""

    train_requested = pyqtSignal()
    probe_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._baseline = float("nan")

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_hyper_group())
        inner.addWidget(self._build_run_group())
        inner.addWidget(self._build_log_group())

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

    def _build_hyper_group(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        grid.addWidget(self.epochs_spin, 0, 1)

        grid.addWidget(QLabel("Batch size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        grid.addWidget(self.batch_spin, 1, 1)

        self.early_check = QCheckBox("Early stopping (val loss)")
        grid.addWidget(self.early_check, 2, 0, 1, 2)
        grid.addWidget(
            hint(
                f"At the default size all three tasks are solved by about "
                f"{DEFAULT_EPOCHS} epochs whenever they can be solved at all. "
                "A model stuck below that is usually on a plateau, not short of "
                "time - more epochs did not move it."
            ),
            3, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("Primary")
        self.train_button.setMinimumHeight(32)
        self.train_button.clicked.connect(self.train_requested.emit)
        row.addWidget(self.train_button, 2)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.setMinimumHeight(32)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.stop_button, 1)
        layout.addLayout(row)

        # Short for the same reason as the sequence workspace's probe button:
        # its text is its minimum width. The detail lives in the tooltip.
        self.probe_button = QPushButton("Does position matter?")
        self.probe_button.setMinimumHeight(30)
        self.probe_button.setToolTip(
            "Trains the same transformer twice from the same weights - once "
            "with positional encoding, once without."
        )
        self.probe_button.clicked.connect(self.probe_requested.emit)
        layout.addWidget(self.probe_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        metrics = QGridLayout()
        self.epoch_label = QLabel("-")
        self.score_label = QLabel("-")
        self.floor_label = QLabel("-")
        self.versus_label = QLabel("-")
        for column, (caption, widget) in enumerate(
            (("Epoch", self.epoch_label), ("Val accuracy", self.score_label),
             ("Giving up", self.floor_label), ("vs giving up", self.versus_label))
        ):
            small = QLabel(caption)
            small.setObjectName("Hint")
            metrics.addWidget(small, 0, column)
            widget.setObjectName("Metric")
            metrics.addWidget(widget, 1, column)
        layout.addLayout(metrics)

        self.baseline_label = QLabel("")
        self.baseline_label.setObjectName("Hint")
        self.baseline_label.setWordWrap(True)
        layout.addWidget(self.baseline_label)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(4000)
        self.log.setMinimumHeight(200)
        layout.addWidget(self.log)
        return group

    # ------------------------------------------------------------------ public

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def clear_log(self) -> None:
        self.log.clear()

    def set_running(self, running: bool) -> None:
        for widget in (self.train_button, self.probe_button, self.epochs_spin,
                       self.batch_spin, self.early_check):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def set_baseline(self, value: float, explain: str = "") -> None:
        self._baseline = float(value)
        if value != value:
            self.floor_label.setText("-")
            self.baseline_label.setText("")
            return
        self.floor_label.setText(f"{value:.3f}")
        self.baseline_label.setText(explain)

    def reset_metrics(self) -> None:
        # floor_label describes the dataset, not the run, so it survives.
        for label in (self.epoch_label, self.score_label, self.versus_label):
            label.setText("-")
        self.progress.setValue(0)

    def show_progress(self, done: int, total: int, epoch: int, logs: dict,
                      label: str) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        self.epoch_label.setText(f"{label} {epoch}".strip())
        score = logs.get("val_accuracy", logs.get("accuracy"))
        if score is None:
            return
        self.score_label.setText(f"{score:.3f}")
        if self._baseline == self._baseline:
            points = 100 * (score - self._baseline)
            self.versus_label.setText(f"{points:+.1f} pts")
