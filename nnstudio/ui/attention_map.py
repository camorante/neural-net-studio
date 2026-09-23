"""The attention heatmap for one sequence, drawn with the tokens on both axes.

Row i is "standing at position i, how much weight went to each column". The
tokens are written on both edges because a heatmap without them is a pretty
pattern and nothing more: the reason to look at attention at all is to check it
against where the answer actually lives.

Two columns get marked. The one the model weighed most is outlined in the
accent colour, and the one the task says the answer depends on is marked in
green. When they coincide, the map is at least pointing the right way - but
only the faithfulness test on the Inspect tab can say whether that pointing is
what produced the answer.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from . import theme


class AttentionMap(QWidget):
    """A square grid of weights, one sequence, one block, one head or their mean."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._weights: np.ndarray | None = None
        self._tokens: list = []
        self._caption = ""
        self._focus = -1
        self._expected = -1
        self.setMinimumHeight(300)

    @property
    def caption(self) -> str:
        return self._caption

    @property
    def focus(self) -> int:
        return self._focus

    def clear(self) -> None:
        self._weights = None
        self._tokens = []
        self._caption = ""
        self._focus = self._expected = -1
        self.update()

    def set_map(self, weights: np.ndarray, tokens: list, caption: str = "",
                expected: int = -1) -> None:
        """`weights` is (length, length) with rows summing to 1."""
        self._weights = np.asarray(weights, dtype="float32")
        self._tokens = [str(t) for t in tokens]
        self._caption = caption
        self._expected = int(expected)
        self._focus = int(np.argmax(self._weights.mean(axis=0)))
        self.update()

    # ------------------------------------------------------------------ paint

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(theme.SURFACE_ALT)))
        painter.drawRoundedRect(rect, 12, 12)

        if self._weights is None or not len(self._tokens):
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Train a model, then pick a sequence on the Inspect tab")
            return

        inner = rect.adjusted(16, 12, -16, -12)
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT)))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), 16),
                         Qt.AlignmentFlag.AlignLeft, self._caption)

        n = len(self._tokens)
        label_w, label_h = 58.0, 34.0
        top = inner.top() + 22 + label_h
        left = inner.left() + label_w
        side = max(40.0, min(inner.width() - label_w - 150,
                             inner.bottom() - top - 26))
        cell = side / n

        small = QFont(self.font())
        small.setPointSize(8)
        painter.setFont(small)

        peak = float(self._weights.max()) or 1.0
        accent = QColor(theme.ACCENT)
        for row in range(n):
            for col in range(n):
                value = float(self._weights[row, col]) / peak
                colour = QColor(accent)
                colour.setAlpha(int(18 + 237 * value))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(colour))
                painter.drawRect(QRectF(left + col * cell, top + row * cell,
                                        cell - 1, cell - 1))

        # The keys along the top are what gets looked AT; the queries down the
        # side are where the looking is done from.
        for index, token in enumerate(self._tokens):
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(QRectF(left + index * cell, top - label_h + 2, cell, 16),
                             Qt.AlignmentFlag.AlignCenter, token)
            painter.drawText(QRectF(left + index * cell, top - 18, cell, 14),
                             Qt.AlignmentFlag.AlignCenter, str(index))
            painter.drawText(QRectF(inner.left(), top + index * cell, label_w - 6, cell),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{index}  {token}")

        if 0 <= self._expected < n:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(theme.SUCCESS), 2.4))
            painter.drawRect(QRectF(left + self._expected * cell - 1, top - 2,
                                    cell + 1, side + 3))
        if 0 <= self._focus < n:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor(theme.OUTPUT_COLOR), 1.8)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(left + self._focus * cell + 2, top + 1,
                                    cell - 5, side - 3))

        legend_x = left + side + 18
        painter.setFont(small)
        for offset, (colour, text) in enumerate((
            (theme.OUTPUT_COLOR, "most attended column"),
            (theme.SUCCESS, "where the answer lives"),
            (theme.ACCENT, "darker = more weight"),
        )):
            y = top + offset * 22
            painter.setPen(QPen(QColor(colour), 2.0))
            painter.drawLine(int(legend_x), int(y + 7), int(legend_x + 16), int(y + 7))
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(QRectF(legend_x + 22, y, 140, 16),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             text)
        painter.drawText(
            QRectF(legend_x, top + 78, 140, 80),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            "Rows: where the model stands. Columns: what it looks at.",
        )
