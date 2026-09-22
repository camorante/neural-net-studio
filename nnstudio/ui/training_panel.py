"""Training controls, live metrics and the run log."""
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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .widgets import hint, scrollable


class TrainingPanel(QWidget):
    """Hyperparameters plus start/stop, progress and log."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    export_requested = pyqtSignal()
    crossval_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_hyper_group())
        inner.addWidget(self._build_control_group())
        inner.addWidget(self._build_crossval_group())
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
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(60)
        grid.addWidget(self.epochs_spin, 0, 1)

        grid.addWidget(QLabel("Batch size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(32)
        grid.addWidget(self.batch_spin, 1, 1)

        self.shuffle_check = QCheckBox("Shuffle every epoch")
        self.shuffle_check.setChecked(True)
        grid.addWidget(self.shuffle_check, 2, 0, 1, 2)

        self.early_check = QCheckBox("Early stopping (val loss)")
        self.early_check.toggled.connect(lambda on: self.patience_spin.setEnabled(on))
        grid.addWidget(self.early_check, 3, 0, 1, 2)

        grid.addWidget(QLabel("Patience:"), 4, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 200)
        self.patience_spin.setValue(10)
        self.patience_spin.setEnabled(False)
        grid.addWidget(self.patience_spin, 4, 1)

        grid.addWidget(
            hint(
                "Small batches update weights more often and generalise better; "
                "large batches run faster per epoch."
            ),
            5,
            0,
            1,
            2,
        )
        return group

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        buttons = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("Primary")
        self.train_button.clicked.connect(self.start_requested.emit)
        buttons.addWidget(self.train_button, 2)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        buttons.addWidget(self.stop_button, 1)

        self.export_button = QPushButton("Save model")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_requested.emit)
        buttons.addWidget(self.export_button, 1)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        metrics = QGridLayout()
        self.epoch_label = QLabel("-")
        self.loss_label = QLabel("-")
        self.val_loss_label = QLabel("-")
        self.metric_label = QLabel("-")
        for column, (caption, widget) in enumerate(
            (
                ("Epoch", self.epoch_label),
                ("Loss", self.loss_label),
                ("Val loss", self.val_loss_label),
                ("Val metric", self.metric_label),
            )
        ):
            caption_label = QLabel(caption)
            caption_label.setObjectName("Hint")
            metrics.addWidget(caption_label, 0, column)
            widget.setObjectName("Metric")
            metrics.addWidget(widget, 1, column)
        layout.addLayout(metrics)
        return group

    def _build_crossval_group(self) -> QGroupBox:
        group = QGroupBox("Cross-validation")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Folds (k):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 10)
        self.k_spin.setValue(5)
        self.k_spin.setMaximumWidth(70)
        row.addWidget(self.k_spin)
        row.addStretch(1)
        self.crossval_button = QPushButton("Cross-validate")
        self.crossval_button.clicked.connect(self.crossval_requested.emit)
        row.addWidget(self.crossval_button, 1)
        layout.addLayout(row)

        self.crossval_result = QLabel("-")
        self.crossval_result.setObjectName("Metric")
        self.crossval_result.setWordWrap(True)
        self.crossval_result.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.crossval_result)

        layout.addWidget(
            hint(
                "Trains k models and throws them all away - what you keep is the "
                "mean and the spread. Use it when one validation split is too "
                "small to trust. Costs k times a normal run."
            )
        )
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(4000)
        self.log.setMinimumHeight(160)
        layout.addWidget(self.log)
        return group

    # ------------------------------------------------------------------ public

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def clear_log(self) -> None:
        self.log.clear()

    def set_running(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.crossval_button.setEnabled(not running)
        self.k_spin.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.early_check.setEnabled(not running)
        self.patience_spin.setEnabled(not running and self.early_check.isChecked())
        self.shuffle_check.setEnabled(not running)
        if running:
            self.progress.setValue(0)

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)

    def set_cross_validating(self, running: bool) -> None:
        """Same lockout as training, but Stop still has to work."""
        self.set_running(running)
        self.stop_button.setEnabled(running)
        if running:
            self.crossval_result.setText("running...")

    def show_fold(self, index: int, total: int, entry: dict) -> None:
        self.progress.setValue(int(index / max(total, 1) * 100))
        self.epoch_label.setText(f"fold {index}/{total}")
        scores = entry.get("scores", {})
        self.loss_label.setText(f"{scores['loss']:.4f}" if "loss" in scores else "-")
        self.val_loss_label.setText("-")
        metric = next((k for k in ("accuracy", "mae", "mse") if k in scores), None)
        self.metric_label.setText(f"{scores[metric]:.4f}" if metric else "-")

    def show_crossval_summary(self, text: str) -> None:
        self.crossval_result.setText(text)

    def show_epoch(self, epoch: int, total: int, logs: dict) -> None:
        self.progress.setValue(int(epoch / max(total, 1) * 100))
        self.epoch_label.setText(f"{epoch}/{total}")
        self.loss_label.setText(f"{logs.get('loss', float('nan')):.4f}")
        self.val_loss_label.setText(
            f"{logs['val_loss']:.4f}" if "val_loss" in logs else "-"
        )
        metric_key = next(
            (k for k in ("val_accuracy", "val_mae", "accuracy", "mae") if k in logs),
            None,
        )
        self.metric_label.setText(f"{logs[metric_key]:.4f}" if metric_key else "-")

    def reset_metrics(self) -> None:
        for label in (
            self.epoch_label,
            self.loss_label,
            self.val_loss_label,
            self.metric_label,
        ):
            label.setText("-")
        self.progress.setValue(0)
