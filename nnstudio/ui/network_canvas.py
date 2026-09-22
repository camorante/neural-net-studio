"""Live diagram of the configured network, drawn with QPainter."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

from . import theme

MAX_VISIBLE_NEURONS = 12
MAX_DRAWN_EDGES = 420


class NetworkCanvas(QWidget):
    """Draws layer columns, their connections, and a pulse while training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list = []
        self._phase = 0.0
        self._training = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)
        self.setMinimumHeight(240)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    # ------------------------------------------------------------------ state

    def set_columns(self, columns: list) -> None:
        self._columns = list(columns or [])
        self.update()

    def set_training(self, active: bool) -> None:
        self._training = bool(active)
        if active:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.018) % 1.0
        self.update()

    # ----------------------------------------------------------------- layout

    def _geometry(self, rect: QRectF) -> list:
        count = len(self._columns)
        if count == 0:
            return []
        usable_height = max(60.0, rect.height() - 84.0)
        centre_y = rect.top() + usable_height / 2.0 + 12.0
        placed = []
        for index, column in enumerate(self._columns):
            units = max(1, int(column["units"]))
            visible = min(units, MAX_VISIBLE_NEURONS)
            step = min(32.0, usable_height / max(visible, 1))
            radius = max(4.0, min(12.0, step * 0.32))
            x = rect.left() + rect.width() * (index + 0.5) / count
            ys = [centre_y + (i - (visible - 1) / 2.0) * step for i in range(visible)]
            placed.append(
                {
                    "x": x,
                    "ys": ys,
                    "radius": radius,
                    "column": column,
                    "hidden_count": units - visible,
                }
            )
        return placed

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
                "Configure the architecture to see the network here",
            )
            return

        inner = rect.adjusted(28, 18, -28, -18)
        placed = self._geometry(inner)
        self._draw_edges(painter, placed)
        self._draw_neurons(painter, placed)
        self._draw_labels(painter, placed, inner)

    def _draw_edges(self, painter: QPainter, placed: list) -> None:
        for left, right in zip(placed, placed[1:]):
            pairs = [(a, b) for a in left["ys"] for b in right["ys"]]
            stride = max(1, math.ceil(len(pairs) / MAX_DRAWN_EDGES))
            pairs = pairs[::stride]
            colour = QColor(theme.BORDER_STRONG)
            colour.setAlpha(110 if len(pairs) < 120 else 70)
            painter.setPen(QPen(colour, 1.0))
            for y_a, y_b in pairs:
                painter.drawLine(QPointF(left["x"], y_a), QPointF(right["x"], y_b))

            if self._training:
                self._draw_pulses(painter, left, right, pairs)

    def _draw_pulses(self, painter: QPainter, left: dict, right: dict, pairs: list) -> None:
        if not pairs:
            return
        glow = QColor(theme.ACCENT)
        glow.setAlpha(220)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        step = max(1, len(pairs) // 14)
        for offset, (y_a, y_b) in enumerate(pairs[::step]):
            t = (self._phase + offset * 0.11) % 1.0
            x = left["x"] + (right["x"] - left["x"]) * t
            y = y_a + (y_b - y_a) * t
            painter.drawEllipse(QPointF(x, y), 2.4, 2.4)

    def _draw_neurons(self, painter: QPainter, placed: list) -> None:
        for entry in placed:
            base = self._colour(entry["column"]["kind"])
            radius = entry["radius"]
            for y in entry["ys"]:
                centre = QPointF(entry["x"], y)
                gradient = QRadialGradient(centre, radius * 1.9)
                bright = QColor(base).lighter(135)
                gradient.setColorAt(0.0, bright)
                gradient.setColorAt(0.55, base)
                halo = QColor(base)
                halo.setAlpha(0)
                gradient.setColorAt(1.0, halo)
                painter.setBrush(QBrush(gradient))
                painter.setPen(QPen(QColor(base).darker(160), 1.0))
                painter.drawEllipse(centre, radius, radius)

            if entry["hidden_count"] > 0:
                painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(
                    QRectF(entry["x"] - 30, entry["ys"][-1] + radius + 3, 60, 14),
                    Qt.AlignmentFlag.AlignCenter,
                    f"+{entry['hidden_count']} more",
                )

    def _draw_labels(self, painter: QPainter, placed: list, inner: QRectF) -> None:
        title_font = QFont(self.font())
        title_font.setPointSize(9)
        title_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(8)

        # Labels must never bleed into the neighbouring column, so the box
        # shrinks with the available slot and the text is elided to match.
        slot = inner.width() / max(len(placed), 1)
        width = max(52.0, min(156.0, slot - 6.0))
        if width < 96:
            title_font.setPointSize(8)
            small_font.setPointSize(7)

        baseline = inner.bottom() - 34
        for entry in placed:
            column = entry["column"]
            left = entry["x"] - width / 2.0

            painter.setFont(title_font)
            painter.setPen(QPen(self._colour(column["kind"])))
            title = f"{column['name']} - {column['units']}"
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(left, baseline, width, 16),
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(title, Qt.TextElideMode.ElideRight, int(width)),
            )

            details = []
            if column.get("activation") and column["activation"] != "-":
                details.append(column["activation"])
            if column.get("extras"):
                details.append(column["extras"])
            painter.setFont(small_font)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(left, baseline + 16, width, 14),
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(
                    " | ".join(details), Qt.TextElideMode.ElideRight, int(width)
                ),
            )
