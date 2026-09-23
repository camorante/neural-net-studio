"""Transformer stage 4: look at one sequence's attention, and at the verdict.

The verdict sits above the map controls on purpose. A student who scrolls to
the heatmap first will read it as an explanation; one who has just been told
"this map is decoration" will read it as the image of weights it is.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import compact_combo, hint, scrollable

AVERAGE = -1


class TransformerInspectPanel(QWidget):
    """Choose which sequence, block and head to draw."""

    show_requested = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_verdict_group())
        inner.addWidget(self._build_map_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self.set_ready(False)

    def _build_verdict_group(self) -> QGroupBox:
        group = QGroupBox("Is the map worth reading?")
        layout = QVBoxLayout(group)
        self.verdict_label = QLabel("Train a model first.")
        self.verdict_label.setObjectName("Metric")
        self.verdict_label.setWordWrap(True)
        layout.addWidget(self.verdict_label)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Hint")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        return group

    def _build_map_group(self) -> QGroupBox:
        group = QGroupBox("Attention map")
        grid = QGridLayout(group)

        # Only offered after the position probe: the arm WITHOUT positions is
        # the one whose map looks focused and turns out to be decoration, and
        # it is worth being able to look at it directly.
        self.arm_label = QLabel("Model:")
        grid.addWidget(self.arm_label, 0, 0)
        self.arm_combo = compact_combo(QComboBox(), 12)
        self.arm_combo.currentIndexChanged.connect(
            lambda _: self._emit() if self.show_button.isEnabled() else None)
        grid.addWidget(self.arm_combo, 0, 1)

        grid.addWidget(QLabel("Sequence:"), 1, 0)
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 0)
        grid.addWidget(self.index_spin, 1, 1)

        grid.addWidget(QLabel("Block:"), 2, 0)
        self.block_spin = QSpinBox()
        self.block_spin.setRange(1, 1)
        grid.addWidget(self.block_spin, 2, 1)

        grid.addWidget(QLabel("Head:"), 3, 0)
        self.head_combo = compact_combo(QComboBox(), 12)
        grid.addWidget(self.head_combo, 3, 1)

        row = QHBoxLayout()
        self.show_button = QPushButton("Show")
        self.show_button.setObjectName("Primary")
        self.show_button.setMinimumHeight(32)
        self.show_button.clicked.connect(self._emit)
        row.addWidget(self.show_button, 1)
        self.next_button = QPushButton("Next sequence")
        self.next_button.setMinimumHeight(32)
        self.next_button.clicked.connect(self._next)
        row.addWidget(self.next_button, 1)
        grid.addLayout(row, 4, 0, 1, 2)

        self.answer_label = QLabel("")
        self.answer_label.setObjectName("Hint")
        self.answer_label.setWordWrap(True)
        grid.addWidget(self.answer_label, 5, 0, 1, 2)

        grid.addWidget(
            hint(
                "Try the last block first - it is the one nearest the answer, "
                "and the one the verdict above was measured on. On the 'after "
                "the cue' task the first block's map is close to noise; the "
                "second one moves with the cue."
            ),
            6, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return group

    # ------------------------------------------------------------------ public

    def set_ready(self, ready: bool, n_sequences: int = 0, n_blocks: int = 1,
                  n_heads: int = 1, arms: tuple = ()) -> None:
        for widget in (self.index_spin, self.block_spin, self.head_combo,
                       self.show_button, self.next_button, self.arm_combo):
            widget.setEnabled(ready)
        self.arm_combo.blockSignals(True)
        self.arm_combo.clear()
        for arm in arms:
            self.arm_combo.addItem(str(arm))
        self.arm_combo.blockSignals(False)
        several = len(arms) > 1
        self.arm_combo.setVisible(several)
        self.arm_label.setVisible(several)
        if not ready:
            self.verdict_label.setText("Train a model first.")
            self.detail_label.setText("")
            self.answer_label.setText("")
            return
        self.index_spin.setRange(0, max(0, n_sequences - 1))
        self.block_spin.setRange(1, max(1, n_blocks))
        self.block_spin.setValue(max(1, n_blocks))
        self.head_combo.blockSignals(True)
        self.head_combo.clear()
        self.head_combo.addItem("Average of all heads", AVERAGE)
        for head in range(n_heads):
            self.head_combo.addItem(f"Head {head + 1}", head)
        self.head_combo.blockSignals(False)

    def show_verdict(self, headline: str, detail: str) -> None:
        self.verdict_label.setText(headline)
        self.detail_label.setText(detail)

    def show_answer(self, text: str) -> None:
        self.answer_label.setText(text)

    # ----------------------------------------------------------------- private

    def _emit(self) -> None:
        self.show_requested.emit(max(0, self.arm_combo.currentIndex()),
                                 int(self.index_spin.value()),
                                 int(self.block_spin.value()) - 1,
                                 int(self.head_combo.currentData()
                                     if self.head_combo.count() else AVERAGE))

    def _next(self) -> None:
        top = self.index_spin.maximum()
        self.index_spin.setValue(0 if self.index_spin.value() >= top
                                 else self.index_spin.value() + 1)
        self._emit()
