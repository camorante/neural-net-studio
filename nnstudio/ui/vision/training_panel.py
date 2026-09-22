"""CNN stage 3: hyperparameters, the two run modes, progress and log."""
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

from ..widgets import hint, scrollable


class VisionTrainingPanel(QWidget):
    """Run controls for the convolutional workspace."""

    train_requested = pyqtSignal()
    compare_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

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

    # ------------------------------------------------------------------- build

    def _build_hyper_group(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(15)
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
                "Convolutions are far heavier than dense layers. On CPU, keep the "
                "image count and the depth modest until you know the run fits in "
                "the time you have."
            ),
            3, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("Primary")
        self.train_button.setMinimumHeight(32)
        self.train_button.clicked.connect(self.train_requested.emit)
        layout.addWidget(self.train_button)

        row = QHBoxLayout()
        self.compare_button = QPushButton("Compare skip on/off")
        self.compare_button.setMinimumHeight(30)
        self.compare_button.clicked.connect(self.compare_requested.emit)
        self.compare_button.setToolTip(
            "Trains the same stack twice from the same initial weights, once "
            "with the shortcut and once without."
        )
        row.addWidget(self.compare_button, 2)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.setMinimumHeight(30)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.stop_button, 1)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        metrics = QGridLayout()
        self.epoch_label = QLabel("-")
        self.loss_label = QLabel("-")
        self.acc_label = QLabel("-")
        for column, (caption, widget) in enumerate(
            (("Epoch", self.epoch_label), ("Val loss", self.loss_label),
             ("Val accuracy", self.acc_label))
        ):
            small = QLabel(caption)
            small.setObjectName("Hint")
            metrics.addWidget(small, 0, column)
            widget.setObjectName("Metric")
            metrics.addWidget(widget, 1, column)
        layout.addLayout(metrics)
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
        for widget in (self.train_button, self.compare_button,
                       self.epochs_spin, self.batch_spin, self.early_check):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def set_compare_enabled(self, enabled: bool) -> None:
        self.compare_button.setEnabled(enabled)
        self.compare_button.setToolTip(
            "Trains the same stack twice from the same initial weights, once "
            "with the shortcut and once without."
            if enabled
            else "Only applies to a from-scratch stack, not a pretrained backbone."
        )

    def apply_preset(self, epochs: int, batch: int) -> None:
        self.epochs_spin.setValue(epochs)
        self.batch_spin.setValue(batch)

    def reset_metrics(self) -> None:
        for label in (self.epoch_label, self.loss_label, self.acc_label):
            label.setText("-")
        self.progress.setValue(0)

    def show_progress(self, done: int, total: int, epoch: int, logs: dict,
                      label: str) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        prefix = f"{label} " if label else ""
        self.epoch_label.setText(f"{prefix}{epoch}")
        self.loss_label.setText(
            f"{logs['val_loss']:.4f}" if "val_loss" in logs else "-"
        )
        self.acc_label.setText(
            f"{logs['val_accuracy']:.4f}" if "val_accuracy" in logs else "-"
        )
