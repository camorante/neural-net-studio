"""Design the network: input size, hidden stack, output head, optimiser."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.dataset import TASK_LABELS
from ..core.model_builder import (
    HIDDEN_ACTIVATIONS,
    MODES_BY_TASK,
    OPTIMIZERS,
    OUTPUT_MODES,
    LayerSpec,
    NetworkConfig,
    estimate_params,
    validate,
)
from .widgets import compact_combo, hint, scrollable

MAX_HIDDEN_LAYERS = 12


class LayerRow(QFrame):
    """One hidden layer: units, activation, dropout, batch norm."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, spec: LayerSpec, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.title = QLabel("Hidden 1")
        self.title.setObjectName("CardTitle")
        layout.addWidget(self.title, 0, 0)

        remove = QPushButton("x")
        remove.setObjectName("Remove")
        remove.setFixedWidth(22)
        remove.setToolTip("Remove this layer")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(remove, 0, 1, Qt.AlignmentFlag.AlignRight)

        # Two columns, not four: this row lives in a narrow sidebar, and a
        # four-column grid forces a minimum width the column cannot give.
        layout.addWidget(QLabel("Units"), 1, 0)
        self.units_spin = QSpinBox()
        self.units_spin.setRange(1, 4096)
        self.units_spin.setValue(spec.units)
        self.units_spin.valueChanged.connect(self.changed.emit)
        layout.addWidget(self.units_spin, 1, 1)

        layout.addWidget(QLabel("Activation"), 2, 0)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(list(HIDDEN_ACTIVATIONS))
        compact_combo(self.activation_combo, 10)
        self.activation_combo.setCurrentText(spec.activation)
        self.activation_combo.currentTextChanged.connect(self.changed.emit)
        layout.addWidget(self.activation_combo, 2, 1)

        layout.addWidget(QLabel("Dropout"), 3, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(spec.dropout)
        self.dropout_spin.valueChanged.connect(self.changed.emit)
        layout.addWidget(self.dropout_spin, 3, 1)

        self.batch_norm_check = QCheckBox("Batch norm")
        self.batch_norm_check.setChecked(spec.batch_norm)
        self.batch_norm_check.toggled.connect(self.changed.emit)
        layout.addWidget(self.batch_norm_check, 4, 0, 1, 2)

        layout.setColumnStretch(1, 1)

    def spec(self) -> LayerSpec:
        return LayerSpec(
            units=self.units_spin.value(),
            activation=self.activation_combo.currentText(),
            dropout=float(self.dropout_spin.value()),
            batch_norm=self.batch_norm_check.isChecked(),
        )

    def set_index(self, index: int) -> None:
        self.title.setText(f"Hidden {index}")


class ArchitecturePanel(QWidget):
    """Emits a NetworkConfig every time anything changes."""

    config_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._locked_by_data = False
        self._task = None

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_io_group())
        inner.addWidget(self._build_hidden_group(), 1)
        inner.addWidget(self._build_output_group())
        inner.addWidget(self._build_optimiser_group())

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

        self.add_layer(LayerSpec(units=16, activation="relu"))
        self.add_layer(LayerSpec(units=8, activation="relu"))
        self._emit()

    # ------------------------------------------------------------------- build

    def _build_io_group(self) -> QGroupBox:
        group = QGroupBox("Input layer")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Input neurons:"), 0, 0)
        self.inputs_spin = QSpinBox()
        self.inputs_spin.setRange(1, 100_000)
        self.inputs_spin.setValue(4)
        self.inputs_spin.valueChanged.connect(self._emit)
        grid.addWidget(self.inputs_spin, 0, 1)

        self.io_hint = hint(
            "Prepare a dataset and this locks to the encoded feature count - "
            "one-hot columns included."
        )
        grid.addWidget(self.io_hint, 1, 0, 1, 2)
        return group

    def _build_hidden_group(self) -> QGroupBox:
        group = QGroupBox("Hidden layers")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(260)
        container = QWidget()
        self.rows_layout = QVBoxLayout(container)
        self.rows_layout.setContentsMargins(0, 0, 6, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch(1)
        self.scroll.setWidget(container)
        layout.addWidget(self.scroll, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ Add hidden layer")
        self.add_button.setObjectName("Ghost")
        self.add_button.clicked.connect(lambda: self.add_layer())
        buttons.addWidget(self.add_button, 1)
        clear_button = QPushButton("Clear all")
        clear_button.clicked.connect(self.clear_layers)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Output layer")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Output neurons:"), 0, 0)
        self.outputs_spin = QSpinBox()
        self.outputs_spin.setRange(1, 10_000)
        self.outputs_spin.setValue(3)
        self.outputs_spin.valueChanged.connect(self._emit)
        grid.addWidget(self.outputs_spin, 0, 1)

        grid.addWidget(QLabel("Activation:"), 1, 0)
        self.output_combo = QComboBox()
        for key, mode in OUTPUT_MODES.items():
            self.output_combo.addItem(mode.label, key)
        compact_combo(self.output_combo)
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        grid.addWidget(self.output_combo, 1, 1)

        self.output_hint = hint("")
        grid.addWidget(self.output_hint, 2, 0, 1, 2)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("Warning")
        self.warning_label.setWordWrap(True)
        self.warning_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        grid.addWidget(self.warning_label, 3, 0, 1, 2)
        return group

    def _build_optimiser_group(self) -> QGroupBox:
        group = QGroupBox("Optimisation")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Optimiser:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(list(OPTIMIZERS))
        compact_combo(self.optimizer_combo, 10)
        self.optimizer_combo.currentTextChanged.connect(self._emit)
        grid.addWidget(self.optimizer_combo, 0, 1)

        grid.addWidget(QLabel("Learning rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        self.lr_spin.valueChanged.connect(self._emit)
        grid.addWidget(self.lr_spin, 1, 1)

        grid.addWidget(QLabel("L2 penalty:"), 2, 0)
        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0.0, 0.1)
        self.l2_spin.setDecimals(5)
        self.l2_spin.setSingleStep(0.0001)
        self.l2_spin.setValue(0.0)
        self.l2_spin.valueChanged.connect(self._emit)
        grid.addWidget(self.l2_spin, 2, 1)

        self.params_label = QLabel("-")
        self.params_label.setObjectName("Metric")
        self.params_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        grid.addWidget(self.params_label, 3, 0, 1, 2)
        return group

    # ------------------------------------------------------------ hidden stack

    def add_layer(self, spec: LayerSpec | None = None) -> None:
        if len(self._rows) >= MAX_HIDDEN_LAYERS:
            return
        if spec is None:
            previous = self._rows[-1].spec().units if self._rows else 32
            spec = LayerSpec(units=max(2, previous // 2), activation="relu")
        row = LayerRow(spec)
        row.changed.connect(self._emit)
        row.remove_requested.connect(self.remove_layer)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        self._rows.append(row)
        self._renumber()
        self._emit()

    def remove_layer(self, row: LayerRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._renumber()
        self._emit()

    def clear_layers(self) -> None:
        for row in list(self._rows):
            self.remove_layer(row)

    def _renumber(self) -> None:
        for index, row in enumerate(self._rows, start=1):
            row.set_index(index)
        self.add_button.setEnabled(len(self._rows) < MAX_HIDDEN_LAYERS)

    # ------------------------------------------------------------------- state

    def apply_dataset(self, bundle) -> None:
        """Lock the input/output sizes to whatever the dataset produced."""
        self._locked_by_data = True
        self._task = bundle.task

        for spin, value in ((self.inputs_spin, bundle.n_inputs), (self.outputs_spin, bundle.n_outputs)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.setEnabled(False)
            spin.blockSignals(False)

        allowed = MODES_BY_TASK[bundle.task]
        self.output_combo.blockSignals(True)
        for index in range(self.output_combo.count()):
            key = self.output_combo.itemData(index)
            item = self.output_combo.model().item(index)
            item.setEnabled(key in allowed)
        if self.output_combo.currentData() not in allowed:
            self.output_combo.setCurrentIndex(
                self.output_combo.findData(allowed[0])
            )
        self.output_combo.blockSignals(False)

        self.io_hint.setText(
            f"Locked by the dataset: {bundle.n_inputs} encoded features -> "
            f"{bundle.n_outputs} outputs ({TASK_LABELS[bundle.task].lower()})."
        )
        self._emit()

    def unlock(self) -> None:
        self._locked_by_data = False
        self._task = None
        self.inputs_spin.setEnabled(True)
        self.outputs_spin.setEnabled(True)
        for index in range(self.output_combo.count()):
            self.output_combo.model().item(index).setEnabled(True)

    def config(self) -> NetworkConfig:
        return NetworkConfig(
            n_inputs=self.inputs_spin.value(),
            n_outputs=self.outputs_spin.value(),
            hidden=[row.spec() for row in self._rows],
            output_mode=self.output_combo.currentData() or "softmax",
            optimizer=self.optimizer_combo.currentText(),
            learning_rate=float(self.lr_spin.value()),
            l2=float(self.l2_spin.value()),
        )

    def _on_output_changed(self) -> None:
        key = self.output_combo.currentData()
        if key and not self._locked_by_data:
            mode = OUTPUT_MODES[key]
            if mode.key == "sigmoid":
                self.outputs_spin.setValue(1)
            elif mode.task == "regression":
                self.outputs_spin.setValue(1)
        self._emit()

    def _emit(self) -> None:
        config = self.config()
        mode = config.mode
        self.output_hint.setText(f"{mode.hint}  Loss: {mode.loss}.")
        warnings = validate(config)
        self.warning_label.setText("\n".join(f"- {w}" for w in warnings))
        self.params_label.setText(f"{estimate_params(config):,} trainable parameters")
        self.config_changed.emit(config)
