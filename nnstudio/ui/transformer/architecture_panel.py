"""Transformer stage 2: width, heads, blocks - and the positional switch.

The positional checkbox sits in its own group at the top, not buried among the
sizes. It is the only control here that changes what the model can perceive
rather than how well it perceives it.
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core import transformer as tr
from ..widgets import compact_combo, hint, scrollable


class TransformerArchitecturePanel(QWidget):
    """The shape of the tiny transformer."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._length = 12
        self._vocab = 8
        self._n_outputs = 7

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_position_group())
        inner.addWidget(self._build_size_group())
        inner.addWidget(self._build_summary_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self.refresh_summary()

    def _build_position_group(self) -> QGroupBox:
        group = QGroupBox("Position")
        layout = QVBoxLayout(group)
        self.positional_check = QCheckBox("Positional encoding")
        self.positional_check.setChecked(True)
        self.positional_check.toggled.connect(self.config_changed.emit)
        layout.addWidget(self.positional_check)
        layout.addWidget(
            hint(
                "Self-attention on its own treats a sequence as an unordered "
                "set: shuffle the tokens and it computes the same thing. Adding "
                "a learned vector per position is the only reason it can tell "
                "'first' from 'last'. Turn this off and watch which tasks break."
            )
        )
        return group

    def _build_size_group(self) -> QGroupBox:
        group = QGroupBox("Size")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Width (d_model):"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(4, 256)
        self.width_spin.setSingleStep(8)
        self.width_spin.setValue(64)
        self.width_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.width_spin, 0, 1)

        grid.addWidget(QLabel("Heads:"), 1, 0)
        self.heads_spin = QSpinBox()
        self.heads_spin.setRange(1, 8)
        self.heads_spin.setValue(2)
        self.heads_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.heads_spin, 1, 1)

        grid.addWidget(QLabel("Blocks:"), 2, 0)
        self.blocks_spin = QSpinBox()
        self.blocks_spin.setRange(1, 4)
        self.blocks_spin.setValue(2)
        self.blocks_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.blocks_spin, 2, 1)

        grid.addWidget(QLabel("Feed-forward:"), 3, 0)
        self.ff_spin = QSpinBox()
        self.ff_spin.setRange(8, 512)
        self.ff_spin.setSingleStep(16)
        self.ff_spin.setValue(64)
        self.ff_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.ff_spin, 3, 1)

        grid.addWidget(QLabel("Pooling:"), 4, 0)
        self.pool_combo = compact_combo(QComboBox(), 14)
        for pool in tr.POOLS:
            self.pool_combo.addItem(tr.POOL_LABELS[pool], pool)
        self.pool_combo.currentIndexChanged.connect(self.config_changed.emit)
        grid.addWidget(self.pool_combo, 4, 1)

        grid.addWidget(QLabel("Dropout:"), 5, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.6)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.dropout_spin, 5, 1)

        grid.addWidget(QLabel("Learning rate:"), 6, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.05)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.003)
        self.lr_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.lr_spin, 6, 1)

        grid.addWidget(
            hint(
                "The heads split the width between them. Measured on the "
                "'after the cue' task: 64 wide with 2 heads solved it on every "
                "seed; 32 wide with 4 heads - 8 per head - solved it on none. "
                "More heads is not free."
            ),
            7, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Summary")
        layout = QVBoxLayout(group)
        self.params_label = QLabel("-")
        self.head_label = QLabel("-")
        self.floor_label = QLabel("-")
        for caption, widget in (("Parameters", self.params_label),
                                ("Width per head", self.head_label),
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
        return group

    # ------------------------------------------------------------------ public

    def set_data_shape(self, length: int, vocab: int, n_outputs: int) -> None:
        self._length, self._vocab, self._n_outputs = int(length), int(vocab), int(n_outputs)
        self.refresh_summary()
        self.config_changed.emit()

    def set_baseline(self, value: float) -> None:
        self.floor_label.setText("-" if value != value
                                 else f"{value:.3f}  accuracy (higher wins)")

    def current_config(self) -> tr.TransformerConfig:
        return tr.TransformerConfig(
            length=self._length, vocab=self._vocab, n_outputs=self._n_outputs,
            d_model=int(self.width_spin.value()),
            n_heads=int(self.heads_spin.value()),
            n_blocks=int(self.blocks_spin.value()),
            ff_dim=int(self.ff_spin.value()),
            dropout=float(self.dropout_spin.value()),
            positional=self.positional_check.isChecked(),
            pool=self.pool_combo.currentData(),
            learning_rate=float(self.lr_spin.value()),
        )

    def problems(self) -> list:
        return tr.validate(self.current_config())

    def refresh_summary(self) -> None:
        config = self.current_config()
        self.params_label.setText(f"{tr.estimate_params(config):,}")
        self.head_label.setText(f"{config.key_dim} numbers per head")
        self.problem_label.setText("\n".join(tr.validate(config)))

    def set_running(self, running: bool) -> None:
        for widget in (self.positional_check, self.width_spin, self.heads_spin,
                       self.blocks_spin, self.ff_spin, self.pool_combo,
                       self.dropout_spin, self.lr_spin):
            widget.setEnabled(not running)
