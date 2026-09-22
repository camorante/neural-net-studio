"""Diagram of a sequence model, with the state loop drawn only when it exists.

The one thing worth seeing here is the arrow that curls back on itself. A
recurrent layer has it: what the layer computed at step t is fed back in at step
t+1, which is the entire mechanism by which "before" can mean anything. Conv1D
and Dense have no such arrow, and their boxes are drawn without one - so the
difference between the architectures is visible rather than asserted.

The timesteps are drawn as real ticks across the top, one per step in the
window, because "24 steps" as a number tells a student nothing about what the
model is being handed.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from . import theme

KIND_COLOURS = {
    "input": theme.INPUT_COLOR,
    "hidden": theme.ACCENT,
    "output": theme.SUCCESS,
}

MAX_TICKS = 60


class SequenceCanvas(QWidget):
    """The window, the layers, and the state loop when there is one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list = []
        self._caption = ""
        self._note = ""
        self._length = 0
        self._recurrent = False
        self.setMinimumHeight(240)

    def set_architecture(self, columns: list, length: int, recurrent: bool,
                         caption: str = "", note: str = "") -> None:
        self._columns = list(columns or [])
        self._length = int(length)
        self._recurrent = bool(recurrent)
        self._caption = caption
        self._note = note
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

        if not self._columns:
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Configure the model to see its shape here")
            return

        inner = rect.adjusted(18, 14, -18, -16)
        self._draw_caption(painter, inner)
        self._draw_window(painter, inner)
        self._draw_columns(painter, inner)
        self._draw_note(painter, inner)

    def _draw_caption(self, painter: QPainter, area: QRectF) -> None:
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT)))
        painter.drawText(QRectF(area.left(), area.top(), area.width(), 16),
                         Qt.AlignmentFlag.AlignLeft, self._caption or "Sequence model")

    def _draw_window(self, painter: QPainter, area: QRectF) -> None:
        """One tick per timestep, left to right, with the direction of time."""
        if self._length < 1:
            return
        top = area.top() + 26
        left, right = area.left(), area.right()

        shown = min(self._length, MAX_TICKS)
        step = (right - left) / max(1, shown)
        painter.setPen(QPen(QColor(theme.INPUT_COLOR), 1.6))
        for index in range(shown):
            x = left + index * step + step / 2
            painter.drawLine(QPointF(x, top), QPointF(x, top + 12))

        painter.setPen(QPen(QColor(theme.TEXT_MUTED), 1.0))
        painter.drawLine(QPointF(left, top + 16), QPointF(right - 8, top + 16))
        self._arrow(painter, QPointF(right - 10, top + 16),
                    QPointF(right, top + 16), QColor(theme.TEXT_MUTED))

        font = QFont(self.font())
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
        caption = f"{self._length} steps, oldest on the left"
        if self._length > MAX_TICKS:
            caption += f" (showing {MAX_TICKS})"
        painter.drawText(QRectF(left, top + 18, right - left, 14),
                         Qt.AlignmentFlag.AlignLeft, caption)

    def _draw_note(self, painter: QPainter, area: QRectF) -> None:
        if not self._note:
            return
        font = QFont(self.font())
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.WARNING)))
        painter.drawText(
            QRectF(area.left(), area.bottom() - 28, area.width(), 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
            | Qt.TextFlag.TextWordWrap,
            self._note,
        )

    def _draw_columns(self, painter: QPainter, area: QRectF) -> None:
        columns = self._columns
        count = len(columns)
        gap = 14.0
        box_w = max(60.0, (area.width() - gap * (count - 1)) / count)

        top = area.top() + 76
        bottom = area.bottom() - (46 if self._note else 26)
        height = max(44.0, min(74.0, bottom - top))
        centre = top + height / 2

        name_font = QFont(self.font())
        name_font.setPointSize(9)
        name_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(8)

        for index, column in enumerate(columns):
            x = area.left() + index * (box_w + gap)
            box = QRectF(x, centre - height / 2, box_w, height)
            colour = QColor(KIND_COLOURS.get(column.get("kind"), theme.ACCENT))

            fill = QColor(colour)
            fill.setAlpha(44)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 1.5))
            painter.drawRoundedRect(box, 7, 7)

            # The loop is the whole point: only a recurrent layer feeds its own
            # output back in at the next step.
            if self._recurrent and column.get("kind") == "hidden":
                self._state_loop(painter, box, QColor(theme.OUTPUT_COLOR))

            if index < count - 1:
                self._arrow(painter, QPointF(box.right() + 1, centre),
                            QPointF(box.right() + gap - 1, centre),
                            QColor(theme.BORDER_STRONG))

            painter.setFont(name_font)
            painter.setPen(QPen(colour))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(box.left() - 4, box.top() - 17, box.width() + 8, 15),
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(column.get("name", ""),
                                   Qt.TextElideMode.ElideRight, int(box.width() + 8)),
            )

            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(box.left() - 4, box.bottom() + 2, box.width() + 8, 26),
                Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                metrics.elidedText(column.get("detail", ""),
                                   Qt.TextElideMode.ElideRight,
                                   int((box.width() + 8) * 2)),
            )

    @staticmethod
    def _state_loop(painter: QPainter, box: QRectF, colour: QColor) -> None:
        """An arrow leaving the top of the box and coming back into it."""
        radius = 13.0
        cx = box.center().x()
        top = box.top()
        loop = QRectF(cx - radius, top - radius * 1.5, radius * 2, radius * 1.6)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 1.6))
        painter.drawArc(loop, 200 * 16, 280 * 16)
        SequenceCanvas._arrow(
            painter,
            QPointF(cx + radius * 0.5, top - 4),
            QPointF(cx + radius * 0.2, top + 1),
            colour,
        )

    @staticmethod
    def _arrow(painter: QPainter, start: QPointF, end: QPointF, colour: QColor) -> None:
        painter.setPen(QPen(colour, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, end)
        head = QPolygonF([
            QPointF(end.x(), end.y()),
            QPointF(end.x() - 4.5, end.y() - 3.2),
            QPointF(end.x() - 4.5, end.y() + 3.2),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawPolygon(head)
