"""Autoencoder stage 3: run controls, live metrics and the log.

The third metric on screen is the one that matters most here. A validation MSE
of 0.14 means nothing on its own - next to the score for answering every image
with the average, it means the network has not started learning.
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

from ..widgets import hint, scrollable


class AutoencoderTrainingPanel(QWidget):
    """Run controls for the autoencoder workspace."""

    train_requested = pyqtSignal()
    sweep_requested = pyqtSignal()
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

    # ------------------------------------------------------------------- build

    def _build_hyper_group(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(25)
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
                "An autoencoder needs patience before it shows anything. On the "
                "synthetic shapes every latent size sits at the give-up floor "
                "until about epoch 7 and only looks like a picture near 25. "
                "The colour takes until about 90, and only if the waist is "
                "wide enough to afford it at all."
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
        self.sweep_button = QPushButton("Compare latent sizes")
        self.sweep_button.setMinimumHeight(30)
        self.sweep_button.clicked.connect(self.sweep_requested.emit)
        self.sweep_button.setToolTip(
            "Trains the same autoencoder at several latent sizes from identical "
            "initial weights, then stacks the reconstructions so you can read a "
            "column top to bottom."
        )
        row.addWidget(self.sweep_button, 2)

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
        self.floor_label = QLabel("-")
        self.versus_label = QLabel("-")
        # The floor sits beside Val MSE on purpose: those two numbers are the
        # comparison, and a loss with nothing next to it teaches nothing.
        for column, (caption, widget) in enumerate(
            (("Epoch", self.epoch_label), ("Val MSE", self.loss_label),
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
        for widget in (self.train_button, self.sweep_button, self.epochs_spin,
                       self.batch_spin, self.early_check):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def set_baseline(self, value: float) -> None:
        """The MSE of answering every image with the average of the training set.

        Shown as a number of its own, not only inside a sentence: before the
        first epoch every other metric reads "-", and a student looking at the
        panel then has nothing to judge the coming loss against.
        """
        self._baseline = float(value)
        if value != value:  # NaN
            self.floor_label.setText("-")
            self.baseline_label.setText("")
            return
        self.floor_label.setText(f"{value:.5f}")
        self.baseline_label.setText(
            "That is the score for answering every image with the average of "
            "the training set. Until the validation MSE drops below it, nothing "
            "has been learned. It is recomputed for every dataset you build."
        )

    def apply_preset(self, epochs: int, batch: int) -> None:
        self.epochs_spin.setValue(int(epochs))
        self.batch_spin.setValue(int(batch))

    def reset_metrics(self) -> None:
        # floor_label is deliberately left alone: it describes the dataset, not
        # the run, and _start() calls this right after _prepare() computed it.
        for label in (self.epoch_label, self.loss_label, self.versus_label):
            label.setText("-")
        self.progress.setValue(0)

    def show_progress(self, done: int, total: int, epoch: int, logs: dict,
                      label: str) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        prefix = f"{label} " if label else ""
        self.epoch_label.setText(f"{prefix}{epoch}")

        loss = logs.get("val_loss", logs.get("loss"))
        if loss is None:
            self.loss_label.setText("-")
            self.versus_label.setText("-")
            return

        self.loss_label.setText(f"{loss:.5f}")
        if self._baseline != self._baseline or loss <= 0:
            self.versus_label.setText("-")
            return
        factor = self._baseline / loss
        self.versus_label.setText(
            f"{factor:.1f}x better" if factor >= 1.0 else f"{1 / factor:.1f}x WORSE"
        )
