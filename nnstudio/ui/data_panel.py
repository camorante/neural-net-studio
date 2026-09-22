"""Load a dataset, pick the target, and adapt it into tensors."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import dataset as ds
from .widgets import compact_combo, hint

PREVIEW_ROWS = 80
PREVIEW_COLUMNS = 30

TASK_CHOICES = (
    ("Auto-detect", None),
    (ds.TASK_LABELS[ds.TASK_BINARY], ds.TASK_BINARY),
    (ds.TASK_LABELS[ds.TASK_MULTICLASS], ds.TASK_MULTICLASS),
    (ds.TASK_LABELS[ds.TASK_REGRESSION], ds.TASK_REGRESSION),
)


class DataPanel(QWidget):
    """Source selection, preview, and the adaptation controls."""

    dataset_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self._source_name = ""
        self._spec: ds.PreparationSpec | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_source_group())
        root.addWidget(self._build_preview_group(), 1)
        root.addWidget(self._build_adapt_group())

        self._load_builtin(ds.BUILTIN_DATASETS[0])

    # ------------------------------------------------------------------- build

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.builtin_combo = QComboBox()
        self.builtin_combo.addItems(list(ds.BUILTIN_DATASETS))
        compact_combo(self.builtin_combo)
        self.builtin_combo.currentTextChanged.connect(self._load_builtin)
        row.addWidget(QLabel("Demo:"))
        row.addWidget(self.builtin_combo, 1)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        load_button = QPushButton("Load file...")
        load_button.clicked.connect(self._choose_file)
        buttons.addWidget(load_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.source_label = QLabel("-")
        self.source_label.setObjectName("Subtle")
        self.source_label.setWordWrap(True)
        self.source_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.source_label)
        layout.addWidget(
            hint("CSV, TSV, TXT, Excel, JSON or Parquet. One row per sample, one column per feature.")
        )
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectColumns)
        layout.addWidget(self.table)
        return group

    def _build_adapt_group(self) -> QGroupBox:
        group = QGroupBox("Adaptation")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Target column:"), 0, 0)
        self.target_combo = QComboBox()
        compact_combo(self.target_combo)
        self.target_combo.currentTextChanged.connect(self._refresh_task_preview)
        grid.addWidget(self.target_combo, 0, 1)

        grid.addWidget(QLabel("Problem type:"), 1, 0)
        self.task_combo = QComboBox()
        for label, _ in TASK_CHOICES:
            self.task_combo.addItem(label)
        compact_combo(self.task_combo)
        self.task_combo.currentIndexChanged.connect(self._refresh_task_preview)
        grid.addWidget(self.task_combo, 1, 1)

        grid.addWidget(QLabel("Validation split:"), 2, 0)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.05, 0.5)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(0.2)
        self.split_spin.setDecimals(2)
        grid.addWidget(self.split_spin, 2, 1)

        grid.addWidget(QLabel("Random seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(42)
        grid.addWidget(self.seed_spin, 3, 1)

        self.scale_features_check = QCheckBox("Standardise numeric features")
        self.scale_features_check.setChecked(True)
        grid.addWidget(self.scale_features_check, 4, 0, 1, 2)

        self.scale_target_check = QCheckBox("Standardise the target")
        self.scale_target_check.setToolTip(
            "Leave this off when the output layer is ReLU or Softplus: "
            "standardising makes the target negative and those activations cannot reach it."
        )
        grid.addWidget(self.scale_target_check, 5, 0, 1, 2)

        self.prepare_button = QPushButton("Prepare data")
        self.prepare_button.setObjectName("Primary")
        self.prepare_button.clicked.connect(self._prepare)
        grid.addWidget(self.prepare_button, 6, 0, 1, 2)

        self.summary_label = QLabel("-")
        self.summary_label.setObjectName("Hint")
        self.summary_label.setWordWrap(True)
        self.summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        grid.addWidget(self.summary_label, 7, 0, 1, 2)
        return group

    # ----------------------------------------------------------------- loading

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", ds.FILE_FILTER)
        if not path:
            return
        try:
            frame = ds.load_file(path)
        except ds.DatasetError as exc:
            QMessageBox.warning(self, "Could not load the file", str(exc))
            return
        self.builtin_combo.blockSignals(True)
        self.builtin_combo.setCurrentIndex(-1)
        self.builtin_combo.blockSignals(False)
        self._set_frame(frame, Path(path).name, suggested_target=frame.columns[-1])

    def _load_builtin(self, label: str) -> None:
        if not label:
            return
        try:
            frame, target = ds.load_builtin(label)
        except ds.DatasetError as exc:
            QMessageBox.warning(self, "Could not load the dataset", str(exc))
            return
        self._set_frame(frame, label.split(" - ")[0], suggested_target=target)

    def _set_frame(self, frame: pd.DataFrame, name: str, suggested_target: str) -> None:
        self._frame = frame
        self._source_name = name
        self.source_label.setText(
            f"{name} - {len(frame):,} rows x {frame.shape[1]} columns"
        )

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems([str(c) for c in frame.columns])
        if suggested_target in frame.columns:
            self.target_combo.setCurrentText(str(suggested_target))
        self.target_combo.blockSignals(False)

        self._fill_preview(frame)
        self._refresh_task_preview()
        self.status.emit(f"Loaded {name}")

    def _fill_preview(self, frame: pd.DataFrame) -> None:
        view = frame.iloc[:PREVIEW_ROWS, :PREVIEW_COLUMNS]
        self.table.clear()
        self.table.setRowCount(len(view))
        self.table.setColumnCount(view.shape[1])
        self.table.setHorizontalHeaderLabels([str(c) for c in view.columns])
        for r in range(len(view)):
            for c in range(view.shape[1]):
                value = view.iat[r, c]
                text = "" if pd.isna(value) else (
                    f"{value:.4g}" if isinstance(value, float) else str(value)
                )
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                )
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    # -------------------------------------------------------------- adaptation

    def _selected_task(self):
        return TASK_CHOICES[self.task_combo.currentIndex()][1]

    def _refresh_task_preview(self) -> None:
        if self._frame is None or not self.target_combo.currentText():
            return
        target = self.target_combo.currentText()
        try:
            task = self._selected_task() or ds.infer_task(self._frame[target])
        except ds.DatasetError as exc:
            self.summary_label.setText(str(exc))
            return
        distinct = int(self._frame[target].nunique())
        self.summary_label.setText(
            f"Target '{target}' looks like {ds.TASK_LABELS[task].lower()} "
            f"({distinct} distinct values). Press Prepare data to adapt it."
        )
        self.scale_target_check.setEnabled(task == ds.TASK_REGRESSION)

    def current_spec(self):
        """The settings the loaded bundle was built with, for reuse in k-fold."""
        return self._spec

    def _prepare(self) -> None:
        if self._frame is None:
            return
        spec = ds.PreparationSpec(
            frame=self._frame,
            target_column=self.target_combo.currentText(),
            name=self._source_name,
            task=self._selected_task(),
            val_split=self.split_spin.value(),
            scale_features=self.scale_features_check.isChecked(),
            scale_target=(
                self.scale_target_check.isChecked()
                and self.scale_target_check.isEnabled()
            ),
            seed=self.seed_spin.value(),
        )
        try:
            bundle = spec.prepare()
        except ds.DatasetError as exc:
            QMessageBox.warning(self, "Could not adapt the dataset", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Unexpected error", str(exc))
            return

        parts = [
            f"{ds.TASK_LABELS[bundle.task]}.",
            f"{bundle.n_train} training rows, {bundle.n_val} validation rows.",
            f"{bundle.n_inputs} input neurons after encoding, {bundle.n_outputs} output neurons.",
        ]
        if bundle.class_names:
            shown = ", ".join(bundle.class_names[:6])
            more = "..." if len(bundle.class_names) > 6 else ""
            parts.append(f"Classes: {shown}{more}")
        if bundle.dropped_columns:
            parts.append(
                "Dropped high-cardinality columns: " + ", ".join(bundle.dropped_columns)
            )
        self.summary_label.setText(" ".join(parts))
        self._spec = spec
        self.dataset_ready.emit(bundle)
        self.status.emit(f"{bundle.name} adapted - {bundle.n_inputs} inputs")
