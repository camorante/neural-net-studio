"""Diagram of the convolutional stack, plus one residual block expanded.

The top row is the stage chain. The bottom row opens a single block so the skip
connection stops being a word and becomes a visible arc that bypasses the
convolutions. With the shortcut switched off the arc disappears and the label
says so - which is the whole lesson.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from . import theme

BLOCK_STEPS = ("Conv 3x3", "BatchNorm", "ReLU", "Conv 3x3", "BatchNorm")


class ResNetCanvas(QWidget):
    """Stage chain on top, one expanded block underneath."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list = []
        self._use_skip = True
        self._depth = 0
        self._caption = ""
        self.setMinimumHeight(300)

    def set_architecture(
        self, columns: list, use_skip: bool, depth: int, caption: str = ""
    ) -> None:
        self._columns = list(columns or [])
        self._use_skip = bool(use_skip)
        self._depth = int(depth)
        self._caption = caption
        self.update()

    @staticmethod
    def _colour(kind: str) -> QColor:
        return {
            "input": QColor(theme.INPUT_COLOR),
            "hidden": QColor(theme.HIDDEN_COLOR),
            "output": QColor(theme.OUTPUT_COLOR),
        }.get(kind, QColor(theme.ACCENT))

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
                "Configure the convolutional stack to see it here",
            )
            return

        inner = rect.adjusted(18, 14, -18, -14)
        split = inner.top() + inner.height() * 0.42
        self._draw_chain(painter, QRectF(inner.left(), inner.top(),
                                         inner.width(), split - inner.top()))
        self._draw_block(painter, QRectF(inner.left(), split + 6,
                                         inner.width(), inner.bottom() - split - 6))

    # ------------------------------------------------------------------ chain

    def _draw_chain(self, painter: QPainter, area: QRectF) -> None:
        count = len(self._columns)
        gap = 12.0
        box_w = max(64.0, (area.width() - gap * (count - 1)) / count)
        box_h = min(58.0, area.height() - 34)
        top = area.top() + 18

        title_font = QFont(self.font())
        title_font.setPointSize(9)
        title_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(8)

        for index, column in enumerate(self._columns):
            x = area.left() + index * (box_w + gap)
            box = QRectF(x, top, box_w, box_h)
            colour = self._colour(column["kind"])

            fill = QColor(colour)
            fill.setAlpha(46)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 1.6))
            painter.drawRoundedRect(box, 8, 8)

            painter.setFont(title_font)
            painter.setPen(QPen(colour))
            painter.drawText(
                QRectF(box.left(), box.top() + 8, box.width(), 14),
                Qt.AlignmentFlag.AlignCenter,
                column["name"],
            )
            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(box.left() + 3, box.top() + 24, box.width() - 6, 26),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                metrics.elidedText(
                    column.get("detail", ""),
                    Qt.TextElideMode.ElideRight,
                    int(box.width() - 6),
                ),
            )

            if index < count - 1:
                self._arrow(
                    painter,
                    QPointF(box.right() + 1, box.center().y()),
                    QPointF(box.right() + gap - 1, box.center().y()),
                    QColor(theme.BORDER_STRONG),
                )

        painter.setFont(small_font)
        painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
        label = f"{self._depth} weight layers on the main path"
        if self._caption:
            label += f"   |   {self._caption}"
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), 14),
            Qt.AlignmentFlag.AlignLeft,
            label,
        )

    # ------------------------------------------------------------------ block

    def _draw_block(self, painter: QPainter, area: QRectF) -> None:
        small_font = QFont(self.font())
        small_font.setPointSize(8)
        header_font = QFont(self.font())
        header_font.setPointSize(9)
        header_font.setBold(True)

        painter.setFont(header_font)
        painter.setPen(QPen(QColor(theme.TEXT)))
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), 16),
            Qt.AlignmentFlag.AlignLeft,
            "Inside one block",
        )

        steps = list(BLOCK_STEPS) + (["Add"] if self._use_skip else []) + ["ReLU"]
        gap = 8.0
        box_w = max(52.0, (area.width() - gap * (len(steps) - 1)) / len(steps))
        box_h = 34.0
        top = area.top() + 52
        boxes = []

        for index, step in enumerate(steps):
            x = area.left() + index * (box_w + gap)
            box = QRectF(x, top, box_w, box_h)
            boxes.append(box)

            is_add = step == "Add"
            colour = QColor(theme.SUCCESS if is_add else theme.HIDDEN_COLOR)
            fill = QColor(colour)
            fill.setAlpha(64 if is_add else 38)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 1.8 if is_add else 1.3))
            painter.drawRoundedRect(box, 7, 7)

            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.TEXT if is_add else theme.TEXT_MUTED)))
            metrics = painter.fontMetrics()
            painter.drawText(
                box,
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(step, Qt.TextElideMode.ElideRight, int(box_w - 4)),
            )

            if index < len(steps) - 1:
                self._arrow(
                    painter,
                    QPointF(box.right() + 1, box.center().y()),
                    QPointF(box.right() + gap - 1, box.center().y()),
                    QColor(theme.BORDER_STRONG),
                )

        if self._use_skip:
            self._draw_skip_arc(painter, boxes, area)
        else:
            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.DANGER)))
            painter.drawText(
                QRectF(area.left(), area.top() + 20, area.width(), 16),
                Qt.AlignmentFlag.AlignLeft,
                "No shortcut: every signal must survive both convolutions. "
                "The block has to learn the whole mapping, not just a correction.",
            )

    def _draw_skip_arc(self, painter: QPainter, boxes: list, area: QRectF) -> None:
        """The arc from the block input to the Add node."""
        add_box = boxes[-2]
        start = QPointF(boxes[0].left() - 4, boxes[0].center().y())
        end = QPointF(add_box.center().x(), add_box.top() - 3)
        peak = min(area.top() + 34, start.y() - 26)

        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x(), peak),
            QPointF(end.x() - 40, peak),
            end,
        )
        pen = QPen(QColor(theme.SUCCESS), 2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawPath(path)

        head = QPolygonF(
            [
                QPointF(end.x(), end.y() + 2),
                QPointF(end.x() - 4.5, end.y() - 6),
                QPointF(end.x() + 4.5, end.y() - 6),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(theme.SUCCESS)))
        painter.drawPolygon(head)

        font = QFont(self.font())
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.SUCCESS)))
        painter.drawText(
            QRectF(area.left(), area.top() + 20, area.width(), 16),
            Qt.AlignmentFlag.AlignLeft,
            "Skip connection: the input is added back, so the block only has to "
            "learn the RESIDUAL - the correction, not the whole mapping.",
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
