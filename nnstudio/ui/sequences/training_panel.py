"""Sequence stage 3: run controls, live metrics and the log.

Three buttons, and the third is the important one. Train answers "how well did
it do"; Compare architectures answers "which layer suits this"; the Order probe
answers "did this ever need a sequence model at all" - and until that one is
answered, the other two are decorating a question nobody checked.
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


class SequenceTrainingPanel(QWidget):
    """Run controls for the sequence workspace."""

    train_requested = pyqtSignal()
    sweep_requested = pyqtSignal()
    probe_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._baseline = float("nan")
        self._lower_is_better = True
        self._metric = "mse"

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
        self.epochs_spin.setValue(20)
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
                "These tasks are small: 20 epochs is enough for every "
                "architecture to separate on the ones that can be learned at "
                "all. If nothing has moved by then, the problem is the task, "
                "not the patience."
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
        self.sweep_button = QPushButton("Compare architectures")
        self.sweep_button.setMinimumHeight(30)
        self.sweep_button.setToolTip(
            "Trains LSTM, GRU, Conv1D and a plain Dense net on the same windows "
            "from the same initial weights."
        )
        self.sweep_button.clicked.connect(self.sweep_requested.emit)
        row.addWidget(self.sweep_button, 2)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.setMinimumHeight(30)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.stop_button, 1)
        layout.addLayout(row)

        self.probe_button = QPushButton("Does order matter? (shuffle probe)")
        self.probe_button.setMinimumHeight(30)
        self.probe_button.setToolTip(
            "Trains the same model twice - once on the real data, once with "
            "every sequence's timesteps permuted. If the scores match, order "
            "carried nothing and a recurrent layer is wasted here."
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
            (("Epoch", self.epoch_label), ("Validation", self.score_label),
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
        for widget in (self.train_button, self.sweep_button, self.probe_button,
                       self.epochs_spin, self.batch_spin, self.early_check):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def set_baseline(self, value: float, metric: str, lower: bool,
                     explain: str = "") -> None:
        """The score for not training at all, shown as a number of its own."""
        self._baseline = float(value)
        self._metric = metric
        self._lower_is_better = bool(lower)
        if value != value:
            self.floor_label.setText("-")
            self.baseline_label.setText("")
            return
        self.floor_label.setText(f"{value:.5f}")
        self.baseline_label.setText(explain)

    def apply_preset(self, epochs: int, batch: int) -> None:
        self.epochs_spin.setValue(int(epochs))
        self.batch_spin.setValue(int(batch))

    def reset_metrics(self) -> None:
        # floor_label is left alone on purpose: it describes the dataset, not
        # the run, and the workspace computes it before training starts.
        for label in (self.epoch_label, self.score_label, self.versus_label):
            label.setText("-")
        self.progress.setValue(0)

    def show_progress(self, done: int, total: int, epoch: int, logs: dict,
                      label: str) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        prefix = f"{label} " if label else ""
        self.epoch_label.setText(f"{prefix}{epoch}")

        key = "val_loss" if self._lower_is_better else "val_accuracy"
        score = logs.get(key, logs.get(key.replace("val_", "")))
        if score is None:
            self.score_label.setText("-")
            self.versus_label.setText("-")
            return

        self.score_label.setText(f"{score:.5f}")
        if self._baseline != self._baseline:
            self.versus_label.setText("-")
            return

        if self._lower_is_better:
            if score <= 0:
                self.versus_label.setText("-")
                return
            factor = self._baseline / score
            self.versus_label.setText(
                f"{factor:.1f}x better" if factor >= 1.0
                else f"{1 / factor:.1f}x WORSE"
            )
        else:
            points = 100 * (score - self._baseline)
            self.versus_label.setText(
                f"+{points:.1f} pts" if points >= 0 else f"{points:.1f} pts"
            )
