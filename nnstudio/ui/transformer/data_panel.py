"""Transformer stage 1: pick a token task and generate it.

Each task's note says what its attention map SHOULD look like if the model
solved it the way a person would. That turns the heatmap on the Inspect tab into
something a student can check against a prediction, rather than a pattern to
admire.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core import transformer as tr
from ..widgets import compact_combo, hint, scrollable


class TransformerDataPanel(QWidget):
    """Generate one of the three synthetic token tasks."""

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

    def _build_task_group(self) -> QGroupBox:
        group = QGroupBox("Task")
        layout = QVBoxLayout(group)
        self.task_combo = compact_combo(QComboBox(), 18)
        for task in tr.TASKS:
            self.task_combo.addItem(tr.TASK_LABELS[task], task)
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
        self.count_spin.setRange(200, 20000)
        self.count_spin.setSingleStep(200)
        self.count_spin.setValue(1600)
        grid.addWidget(self.count_spin, 0, 1)

        grid.addWidget(QLabel("Length:"), 1, 0)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(tr.MIN_LENGTH, tr.MAX_LENGTH)
        self.length_spin.setValue(12)
        self.length_spin.valueChanged.connect(self.plan_changed.emit)
        grid.addWidget(self.length_spin, 1, 1)

        grid.addWidget(QLabel("Vocabulary:"), 2, 0)
        self.vocab_spin = QSpinBox()
        self.vocab_spin.setRange(tr.MIN_VOCAB, tr.MAX_VOCAB)
        self.vocab_spin.setValue(8)
        self.vocab_spin.valueChanged.connect(self.plan_changed.emit)
        grid.addWidget(self.vocab_spin, 2, 1)

        grid.addWidget(QLabel("Seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(7)
        grid.addWidget(self.seed_spin, 3, 1)

        grid.addWidget(
            hint(
                "Token 0 is reserved as the cue marker and never appears as "
                "content, so a vocabulary of 8 gives 7 possible answers. Keep "
                "the length small: the attention map is length x length, and "
                "past about 20 it stops being readable."
            ),
            4, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Build")
        layout = QVBoxLayout(group)
        self.build_button = QPushButton("Generate")
        self.build_button.setObjectName("Primary")
        self.build_button.setMinimumHeight(32)
        self.build_button.clicked.connect(self._emit_build)
        layout.addWidget(self.build_button)

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

    def planned_vocab(self) -> int:
        return int(self.vocab_spin.value())

    def spec(self) -> tr.TokenSpec:
        return tr.TokenSpec(
            task=self.current_task(),
            n_sequences=int(self.count_spin.value()),
            length=int(self.length_spin.value()),
            vocab=int(self.vocab_spin.value()),
            seed=int(self.seed_spin.value()),
        )

    def set_busy(self, busy: bool) -> None:
        for widget in (self.build_button, self.task_combo, self.count_spin,
                       self.length_spin, self.vocab_spin, self.seed_spin):
            widget.setEnabled(not busy)
        if busy:
            self.progress.setValue(0)

    def show_progress(self, done: int, total: int, label: str, speed: float) -> None:
        self.progress.setValue(int(done / max(total, 1) * 100))
        self.summary_label.setText(label)

    def show_summary(self, text: str) -> None:
        self.progress.setValue(100)
        self.summary_label.setText(text)

    # ----------------------------------------------------------------- private

    def _on_task_changed(self) -> None:
        self.task_note.setText(tr.TASK_NOTES.get(self.current_task(), ""))
        self.plan_changed.emit()

    def _emit_build(self) -> None:
        spec = self.spec()

        def builder(on_progress, should_stop):
            return tr.make_tokens(spec, on_progress, should_stop)

        self.build_requested.emit(builder, f"Generating {tr.TASK_LABELS[spec.task]}")
