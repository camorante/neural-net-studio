"""Sequence stage 1: pick a task and generate it.

Two of the four tasks exist to be lost, which is unusual enough to say on the
panel itself rather than in a manual nobody has open. The note under the task
combo is the task's own description, so a student picking "random walk" is told
before they train that persistence cannot be beaten there.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core import sequences as sq
from ..widgets import compact_combo, hint, scrollable


class SequenceDataPanel(QWidget):
    """Generate one of the four synthetic sequence datasets."""

    build_requested = pyqtSignal(object, str)
    plan_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_task_group())
        inner.addWidget(self._build_shape_group())
        inner.addWidget(self._build_run_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self._on_task_changed()

    # ------------------------------------------------------------------- build

    def _build_task_group(self) -> QGroupBox:
        group = QGroupBox("Task")
        layout = QVBoxLayout(group)

        self.task_combo = compact_combo(QComboBox(), 18)
        for task in sq.TASKS:
            self.task_combo.addItem(sq.TASK_LABELS[task], task)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        layout.addWidget(self.task_combo)

        self.task_note = QLabel("")
        self.task_note.setObjectName("Hint")
        self.task_note.setWordWrap(True)
        layout.addWidget(self.task_note)
        return group

    def _build_shape_group(self) -> QGroupBox:
        group = QGroupBox("Shape")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Sequences:"), 0, 0)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(100, 20000)
        self.count_spin.setSingleStep(100)
        self.count_spin.setValue(1200)
        grid.addWidget(self.count_spin, 0, 1)

        grid.addWidget(QLabel("Window length:"), 1, 0)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(sq.MIN_LENGTH, sq.MAX_LENGTH)
        self.length_spin.setValue(24)
        self.length_spin.valueChanged.connect(self.plan_changed.emit)
        grid.addWidget(self.length_spin, 1, 1)

        grid.addWidget(QLabel("Noise:"), 2, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 1.0)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setValue(0.05)
        grid.addWidget(self.noise_spin, 2, 1)

        grid.addWidget(QLabel("Seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(7)
        grid.addWidget(self.seed_spin, 3, 1)

        self.shuffle_check = QCheckBox("Shuffle the timesteps (destroy order)")
        self.shuffle_check.setToolTip(
            "Permutes every sequence's steps independently. The values and their "
            "counts survive; only the arrangement is destroyed."
        )
        grid.addWidget(self.shuffle_check, 4, 0, 1, 2)

        grid.addWidget(
            hint(
                "Longer windows give a recurrent layer more to remember and a "
                "Dense one more weights to spend. The forecasting tasks are "
                "split chronologically - validation is the tail of the series, "
                "never a random sample of it, because a random split would let "
                "the future leak into the past."
            ),
            5, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Build")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.build_button = QPushButton("Generate")
        self.build_button.setObjectName("Primary")
        self.build_button.setMinimumHeight(32)
        self.build_button.clicked.connect(self._emit_build)
        row.addWidget(self.build_button, 2)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.summary_label = QLabel("No dataset yet.")
        self.summary_label.setObjectName("Hint")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        return group

    # ------------------------------------------------------------------ public

    def current_task(self) -> str:
        return self.task_combo.currentData()

    def planned_length(self) -> int:
        return int(self.length_spin.value())

    def spec(self) -> sq.SequenceSpec:
        return sq.SequenceSpec(
            task=self.current_task(),
            n_sequences=int(self.count_spin.value()),
            length=int(self.length_spin.value()),
            noise=float(self.noise_spin.value()),
            seed=int(self.seed_spin.value()),
            shuffle_time=self.shuffle_check.isChecked(),
        )

    def set_busy(self, busy: bool) -> None:
        for widget in (self.build_button, self.task_combo, self.count_spin,
                       self.length_spin, self.noise_spin, self.seed_spin,
                       self.shuffle_check):
            widget.setEnabled(not busy)
        if busy:
            self.progress.setValue(0)

    def show_progress(self, done: int, total: int, label: str, speed: float) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        self.summary_label.setText(label)

    def show_summary(self, text: str) -> None:
        self.progress.setValue(100)
        self.summary_label.setText(text)

    def show_status(self, text: str) -> None:
        self.summary_label.setText(text)

    # ----------------------------------------------------------------- private

    def _on_task_changed(self) -> None:
        task = self.current_task()
        self.task_note.setText(sq.TASK_NOTES.get(task, ""))
        self.plan_changed.emit()

    def _emit_build(self) -> None:
        spec = self.spec()

        def builder(on_progress, should_stop):
            return sq.make_sequences(spec, on_progress, should_stop)

        self.build_requested.emit(builder, f"Generating {sq.TASK_LABELS[spec.task]}")
