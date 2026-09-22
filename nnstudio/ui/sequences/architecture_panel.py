"""Sequence stage 2: choose the architecture.

The floor is shown here as well as on the Training tab, for the same reason it
is in the autoencoder workspace: before the first epoch every other number reads
"-", and a student looking at this panel then has nothing to judge the coming
score against.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core import sequences as sq
from ...core.sequence_trainer import SWEEP_BATCH, SWEEP_EPOCHS
from ..widgets import compact_combo, hint, scrollable


class SequenceArchitecturePanel(QWidget):
    """Which layer reads the window, and how wide it is."""

    config_changed = pyqtSignal()
    preset_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._length = 24
        self._task_kind = sq.FORECAST
        self._n_outputs = 1
        self._baseline = float("nan")

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_kind_group())
        inner.addWidget(self._build_size_group())
        inner.addWidget(self._build_summary_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self._on_kind_changed()

    # ------------------------------------------------------------------- build

    def _build_kind_group(self) -> QGroupBox:
        group = QGroupBox("Architecture")
        layout = QVBoxLayout(group)

        self.kind_combo = compact_combo(QComboBox(), 18)
        for kind in sq.KINDS:
            self.kind_combo.addItem(sq.KIND_LABELS[kind], kind)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        layout.addWidget(self.kind_combo)

        self.kind_note = QLabel("")
        self.kind_note.setObjectName("Hint")
        self.kind_note.setWordWrap(True)
        layout.addWidget(self.kind_note)

        self.sweep_button = QPushButton(
            f"Preset: compare architectures ({SWEEP_EPOCHS} epochs)"
        )
        self.sweep_button.setMinimumHeight(30)
        self.sweep_button.clicked.connect(
            lambda: self.preset_requested.emit(SWEEP_EPOCHS, SWEEP_BATCH)
        )
        layout.addWidget(self.sweep_button)
        return group

    def _build_size_group(self) -> QGroupBox:
        group = QGroupBox("Size")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Units:"), 0, 0)
        self.units_spin = QSpinBox()
        self.units_spin.setRange(1, 512)
        self.units_spin.setValue(32)
        self.units_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.units_spin, 0, 1)

        grid.addWidget(QLabel("Layers:"), 1, 0)
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 4)
        self.layers_spin.setValue(1)
        self.layers_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.layers_spin, 1, 1)

        self.bidirectional_check = QCheckBox("Bidirectional (read it both ways)")
        self.bidirectional_check.setToolTip(
            "Runs a second copy backwards and joins them. Only means anything "
            "for a recurrent layer, and only when you are allowed to see the "
            "whole sequence before answering - never in live forecasting."
        )
        self.bidirectional_check.toggled.connect(self.config_changed.emit)
        grid.addWidget(self.bidirectional_check, 2, 0, 1, 2)

        grid.addWidget(QLabel("Dropout:"), 3, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.dropout_spin, 3, 1)

        grid.addWidget(QLabel("Learning rate:"), 4, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.005)
        self.lr_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.lr_spin, 4, 1)

        grid.setColumnStretch(1, 1)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Summary")
        layout = QVBoxLayout(group)

        self.params_label = QLabel("-")
        self.shape_label = QLabel("-")
        self.floor_label = QLabel("-")
        for caption, widget in (("Parameters (estimate)", self.params_label),
                                ("Window", self.shape_label),
                                ("Score to beat", self.floor_label)):
            small = QLabel(caption)
            small.setObjectName("Hint")
            layout.addWidget(small)
            widget.setObjectName("Metric")
            layout.addWidget(widget)

        self.problem_label = QLabel("")
        self.problem_label.setObjectName("Hint")
        self.problem_label.setWordWrap(True)
        layout.addWidget(self.problem_label)

        layout.addWidget(
            hint(
                "Compare the parameter counts before you train. A recurrent "
                "layer usually costs several times what a Dense one does on the "
                "same window, and it has to earn that back."
            )
        )
        return group

    # ------------------------------------------------------------------ public

    def set_data_shape(self, length: int, task_kind: str, n_outputs: int) -> None:
        self._length = int(length)
        self._task_kind = task_kind
        self._n_outputs = int(n_outputs)
        self.refresh_summary()
        self.config_changed.emit()

    def set_baseline(self, value: float, metric: str = "", lower: bool = True) -> None:
        self._baseline = float(value)
        if value != value:
            self.floor_label.setText("-")
            return
        direction = "lower wins" if lower else "higher wins"
        self.floor_label.setText(f"{value:.5f}  {metric} ({direction})")

    def current_config(self) -> sq.SequenceConfig:
        return sq.SequenceConfig(
            length=self._length,
            n_features=1,
            kind=self.kind_combo.currentData(),
            units=int(self.units_spin.value()),
            layers=int(self.layers_spin.value()),
            bidirectional=self.bidirectional_check.isChecked(),
            dropout=float(self.dropout_spin.value()),
            task_kind=self._task_kind,
            n_outputs=self._n_outputs,
            learning_rate=float(self.lr_spin.value()),
        )

    def problems(self) -> list:
        return sq.validate(self.current_config())

    def refresh_summary(self) -> None:
        config = self.current_config()
        self.params_label.setText(f"{sq.estimate_params(config):,}")
        self.shape_label.setText(
            f"{config.length} steps x {config.n_features} -> "
            + ("next value" if config.task_kind == sq.FORECAST
               else f"{config.n_outputs} classes")
        )
        problems = sq.validate(config)
        self.problem_label.setText("\n".join(problems))

    def set_running(self, running: bool) -> None:
        for widget in (self.kind_combo, self.units_spin, self.layers_spin,
                       self.bidirectional_check, self.dropout_spin, self.lr_spin,
                       self.sweep_button):
            widget.setEnabled(not running)

    # ----------------------------------------------------------------- private

    def _on_kind_changed(self) -> None:
        kind = self.kind_combo.currentData()
        self.kind_note.setText(sq.KIND_NOTES.get(kind, ""))
        recurrent = kind in (sq.LSTM, sq.GRU, sq.RNN)
        self.bidirectional_check.setEnabled(recurrent)
        if not recurrent:
            self.bidirectional_check.setChecked(False)
        self.refresh_summary()
        self.config_changed.emit()
