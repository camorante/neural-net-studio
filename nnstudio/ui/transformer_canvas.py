"""Diagram of the tiny transformer: a row of blocks, and what the embedding carries.

The box worth looking at is the embedding one. Its caption says whether the
positions were added in, because that single choice decides whether the
attention boxes after it can tell one position from another at all. When the
positions are off, the box is drawn in the warning colour - it is the most
consequential switch in the workspace and it should look like one.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from . import theme

KIND_COLOURS = {
    "input": theme.INPUT_COLOR,
    "embed": theme.HIDDEN_COLOR,
    "attention": theme.OUTPUT_COLOR,
    "hidden": theme.ACCENT,
    "output": theme.SUCCESS,
}


class TransformerCanvas(QWidget):
    """A row of boxes: tokens, embedding, attention and feed-forward, answer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list = []
        self._caption = ""
        self._note = ""
        self._positional = True
        self.setMinimumHeight(200)

    def set_architecture(self, columns: list, positional: bool,
                         caption: str = "", note: str = "") -> None:
        self._columns = list(columns or [])
        self._positional = bool(positional)
        self._caption = caption
        self._note = note
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(theme.SURFACE_ALT)))
        painter.drawRoundedRect(rect, 12, 12)
        if not self._columns:
            return

        inner = rect.adjusted(18, 14, -18, -16)
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT)))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), 16),
                         Qt.AlignmentFlag.AlignLeft, self._caption)

        count = len(self._columns)
        gap = 12.0
        box_w = max(58.0, (inner.width() - gap * (count - 1)) / count)
        top = inner.top() + 44
        height = max(40.0, min(64.0, inner.bottom() - top - (58 if self._note else 34)))
        centre = top + height / 2

        name_font = QFont(self.font())
        name_font.setPointSize(9)
        name_font.setBold(True)
        small = QFont(self.font())
        small.setPointSize(8)

        for index, column in enumerate(self._columns):
            x = inner.left() + index * (box_w + gap)
            box = QRectF(x, top, box_w, height)
            kind = column.get("kind")
            colour = QColor(KIND_COLOURS.get(kind, theme.ACCENT))
            if kind == "embed" and not self._positional:
                colour = QColor(theme.DANGER)
            fill = QColor(colour)
            fill.setAlpha(44)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 1.5))
            painter.drawRoundedRect(box, 7, 7)

            if kind == "attention":
                # Every position to every position: the whole idea, in six lines.
                painter.setPen(QPen(colour, 0.9))
                for a in range(3):
                    for b in range(3):
                        painter.drawLine(
                            QPointF(box.left() + 10, box.top() + 12 + a * (height - 24) / 2),
                            QPointF(box.right() - 10, box.top() + 12 + b * (height - 24) / 2),
                        )

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
            painter.setFont(small)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(
                QRectF(box.left() - 4, box.bottom() + 2, box.width() + 8, 28),
                Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                column.get("detail", ""),
            )

        if self._note:
            painter.setFont(small)
            painter.setPen(QPen(QColor(theme.WARNING)))
            painter.drawText(
                QRectF(inner.left(), inner.bottom() - 28, inner.width(), 28),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
                | Qt.TextFlag.TextWordWrap,
                self._note,
            )

    @staticmethod
    def _arrow(painter: QPainter, start: QPointF, end: QPointF, colour: QColor) -> None:
        painter.setPen(QPen(colour, 1.4))
        painter.drawLine(start, end)
        head = QPolygonF([QPointF(end.x(), end.y()),
                          QPointF(end.x() - 4.5, end.y() - 3.2),
                          QPointF(end.x() - 4.5, end.y() + 3.2)])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawPolygon(head)
