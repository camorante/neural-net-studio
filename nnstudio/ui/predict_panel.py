"""Feed raw feature values into the trained model and read the output."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .widgets import hint

MAX_MANUAL_FEATURES = 64


class PredictPanel(QWidget):
    """One editor per raw feature column, plus the model's reading."""

    predict_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle = None
        self._editors: dict = {}
        self._rng = np.random.default_rng()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.info_label = hint(
            "Prepare a dataset and train the network, then try an input here."
        )
        root.addWidget(self.info_label)

        group = QGroupBox("Feature values")
        group_layout = QVBoxLayout(group)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self._container = QWidget()
        self.form = QFormLayout(self._container)
        self.form.setContentsMargins(4, 4, 12, 4)
        self.form.setSpacing(6)
        self.scroll.setWidget(self._container)
        group_layout.addWidget(self.scroll)
        root.addWidget(group, 1)

        buttons = QHBoxLayout()
        self.sample_button = QPushButton("Random validation row")
        self.sample_button.setEnabled(False)
        self.sample_button.clicked.connect(self._fill_random)
        buttons.addWidget(self.sample_button)

        self.reset_button = QPushButton("Reset to medians")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._fill_defaults)
        buttons.addWidget(self.reset_button)

        self.predict_button = QPushButton("Predict")
        self.predict_button.setObjectName("Primary")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self._emit_request)
        buttons.addWidget(self.predict_button)
        root.addLayout(buttons)

        self.truth_label = QLabel("")
        self.truth_label.setObjectName("Hint")
        root.addWidget(self.truth_label)

        self.result_label = QLabel("no prediction yet")
        self.result_label.setObjectName("Result")
        self.result_label.setWordWrap(True)
        root.addWidget(self.result_label)

    # ------------------------------------------------------------------ public

    def set_bundle(self, bundle) -> None:
        self._bundle = bundle
        self._clear_form()
        self.truth_label.setText("")
        self.result_label.setText("no prediction yet")

        columns = bundle.raw_columns
        if len(columns) > MAX_MANUAL_FEATURES:
            self.info_label.setText(
                f"{len(columns)} feature columns - too many to type by hand. "
                "Use 'Random validation row' to sample a real example instead."
            )
        else:
            self.info_label.setText(
                f"Raw values for {len(columns)} columns. They go through the same "
                "imputation, scaling and one-hot encoding used for training."
            )
            for column in columns:
                self.form.addRow(column, self._build_editor(bundle, column))

        self.sample_button.setEnabled(bundle.val_index.size > 0)
        self.reset_button.setEnabled(bool(self._editors))
        self.predict_button.setEnabled(False)

    def set_model_ready(self, ready: bool) -> None:
        self.predict_button.setEnabled(ready and self._bundle is not None)

    def show_result(self, text: str) -> None:
        self.result_label.setText(text)

    # ----------------------------------------------------------------- editors

    def _clear_form(self) -> None:
        self._editors.clear()
        while self.form.rowCount():
            self.form.removeRow(0)

    def _build_editor(self, bundle, column: str) -> QWidget:
        if column in bundle.categorical_columns:
            combo = QComboBox()
            combo.addItems([str(v) for v in bundle.raw_choices.get(column, [])])
            combo.setCurrentText(str(bundle.raw_defaults.get(column, "")))
            self._editors[column] = combo
            return combo

        spin = QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(4)
        spin.setSingleStep(0.1)
        default = bundle.raw_defaults.get(column, 0.0)
        spin.setValue(float(default) if default == default else 0.0)
        self._editors[column] = spin
        return spin

    def _values(self) -> dict:
        values = {}
        for column, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                values[column] = editor.currentText()
            else:
                values[column] = float(editor.value())
        return values

    def _apply(self, values: dict) -> None:
        for column, editor in self._editors.items():
            if column not in values:
                continue
            value = values[column]
            if isinstance(editor, QComboBox):
                editor.setCurrentText(str(value))
            else:
                try:
                    editor.setValue(float(value))
                except (TypeError, ValueError):
                    pass

    # ----------------------------------------------------------------- actions

    def _fill_defaults(self) -> None:
        if self._bundle:
            self._apply(dict(self._bundle.raw_defaults))
            self.truth_label.setText("")

    def _fill_random(self) -> None:
        if not self._bundle:
            return
        values, truth = self._bundle.sample_raw_row(self._rng)
        self._pending = values
        self._apply(values)
        self.truth_label.setText(f"True value for this row: {truth}")
        if not self._editors:
            self.predict_requested.emit(values)

    def _emit_request(self) -> None:
        if not self._bundle:
            return
        values = self._values() if self._editors else getattr(self, "_pending", {})
        if not values:
            self.result_label.setText("Sample a validation row first")
            return
        self.predict_requested.emit(values)
