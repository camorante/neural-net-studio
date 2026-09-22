"""Diagram of the autoencoder, with every box sized by its real tensor width.

The boxes are drawn from measured shapes, not from the tidy hourglass every
textbook prints. That matters, because for a convolutional encoder the tidy
picture is wrong: the first strided stage usually holds MORE numbers than the
image it came from, and the whole squeeze happens in one step, at the dense
layer into the latent code. Drawing the real widths puts that on screen instead
of hiding it behind a pretty taper.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from . import theme

KIND_COLOURS = {
    "input": theme.INPUT_COLOR,
    "encoder": theme.ACCENT,
    "latent": theme.OUTPUT_COLOR,
    "decoder": theme.HIDDEN_COLOR,
    "output": theme.SUCCESS,
}


class AutoencoderCanvas(QWidget):
    """A row of boxes whose heights are the real number of values at each step."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list = []
        self._caption = ""
        self._note = ""
        self.setMinimumHeight(260)

    def set_architecture(self, columns: list, caption: str = "", note: str = "") -> None:
        self._columns = list(columns or [])
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
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "Configure the autoencoder to see its shape here",
            )
            return

        inner = rect.adjusted(18, 14, -18, -16)
        self._draw_caption(painter, inner)
        self._draw_columns(painter, inner)
        self._draw_note(painter, inner)

    def _draw_caption(self, painter: QPainter, area: QRectF) -> None:
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT)))
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), 16),
            Qt.AlignmentFlag.AlignLeft,
            self._caption or "Autoencoder",
        )

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
        gap = 10.0
        box_w = max(52.0, (area.width() - gap * (count - 1)) / count)

        top = area.top() + 26
        bottom = area.bottom() - (44 if self._note else 20)
        band = max(60.0, bottom - top)
        centre = top + band / 2

        # Square root, not linear: the widest box here can hold 250 times the
        # numbers of the narrowest, and a linear map would render the latent
        # code as a hairline nobody can read a label inside.
        values = [max(1, int(c.get("values", 1))) for c in columns]
        scale = band / math.sqrt(max(values))

        name_font = QFont(self.font())
        name_font.setPointSize(9)
        name_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(8)

        boxes = []
        for index, column in enumerate(columns):
            height = max(18.0, math.sqrt(values[index]) * scale)
            x = area.left() + index * (box_w + gap)
            box = QRectF(x, centre - height / 2, box_w, height)
            boxes.append(box)

            colour = QColor(KIND_COLOURS.get(column.get("kind"), theme.ACCENT))
            is_latent = column.get("kind") == "latent"

            fill = QColor(colour)
            fill.setAlpha(80 if is_latent else 44)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 2.2 if is_latent else 1.5))
            painter.drawRoundedRect(box, 7, 7)

            if index < count - 1:
                self._arrow(
                    painter,
                    QPointF(box.right() + 1, centre),
                    QPointF(box.right() + gap - 1, centre),
                    QColor(theme.BORDER_STRONG),
                )

        # Labels go above and below the boxes rather than inside them: the
        # latent box is deliberately the smallest thing on screen.
        for index, (column, box) in enumerate(zip(columns, boxes)):
            colour = QColor(KIND_COLOURS.get(column.get("kind"), theme.ACCENT))
            painter.setFont(name_font)
            painter.setPen(QPen(colour))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(box.left() - 4, box.top() - 17, box.width() + 8, 15),
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(
                    column.get("name", ""),
                    Qt.TextElideMode.ElideRight,
                    int(box.width() + 8),
                ),
            )

            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(box.left() - 4, box.bottom() + 2, box.width() + 8, 14),
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(
                    column.get("detail", ""),
                    Qt.TextElideMode.ElideRight,
                    int(box.width() + 8),
                ),
            )
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(
                QRectF(box.left() - 4, box.bottom() + 15, box.width() + 8, 14),
                Qt.AlignmentFlag.AlignCenter,
                f"{column.get('values', 0):,}",
            )

    @staticmethod
    def _arrow(painter: QPainter, start: QPointF, end: QPointF, colour: QColor) -> None:
        painter.setPen(QPen(colour, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, end)
        head = QPolygonF(
            [
                QPointF(end.x(), end.y()),
                QPointF(end.x() - 4.5, end.y() - 3.2),
                QPointF(end.x() - 4.5, end.y() + 3.2),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawPolygon(head)
