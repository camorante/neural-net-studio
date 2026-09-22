"""Rows of images stacked so a reconstruction sits directly under its original.

This is the autoencoder workspace's main instrument. A loss curve says the
error went down; only these rows say what the error IS. Put latent 2 under
latent 64 under the original and the bottleneck stops being a number - you can
see the shape survive and the edges go.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from . import theme
from .image_grid import to_qimage

LABEL_WIDTH = 96
ROW_GAP = 6
CELL_GAP = 5
CAPTION_HEIGHT = 20
ERROR_HEIGHT = 12


class ReconstructionGrid(QWidget):
    """One labelled row of images per model, aligned column by column."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._caption = ""
        self._placeholder = (
            "Train an autoencoder to see originals and reconstructions here"
        )
        self.setMinimumHeight(260)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._rows = []
        self._caption = ""
        self.update()

    def show_rows(self, rows: list, caption: str = "") -> None:
        """Each row is {label, images, errors?, colour?}.

        The images in every row must be the same pictures in the same order,
        because the whole point is reading a column top to bottom.
        """
        prepared = []
        for row in rows or []:
            images = np.asarray(row.get("images"))
            if images.size == 0:
                continue
            prepared.append(
                {
                    "label": str(row.get("label", "")),
                    "pixmaps": [
                        QPixmap.fromImage(to_qimage(frame)) for frame in images
                    ],
                    "errors": row.get("errors"),
                    "colour": row.get("colour") or theme.BORDER_STRONG,
                }
            )
        self._rows = prepared
        self._caption = caption
        self.update()

    # ------------------------------------------------------------------ paint

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(theme.SURFACE_ALT)))
        painter.drawRoundedRect(rect, 12, 12)

        if not self._rows:
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        inner = rect.adjusted(12, 10, -12, -10)
        if self._caption:
            font = QFont(self.font())
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(
                QRectF(inner.left(), inner.top(), inner.width(), CAPTION_HEIGHT),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._caption,
            )
            inner = inner.adjusted(0, CAPTION_HEIGHT, 0, 0)

        rows = self._rows
        columns = max(len(row["pixmaps"]) for row in rows)
        has_errors = any(row["errors"] is not None for row in rows)
        extra = ERROR_HEIGHT if has_errors else 0

        label_width = min(LABEL_WIDTH, inner.width() * 0.22)
        grid_width = inner.width() - label_width
        cell_w = (grid_width - CELL_GAP * (columns - 1)) / max(1, columns)
        row_h = (inner.height() - ROW_GAP * (len(rows) - 1)) / len(rows)
        side = max(16.0, min(cell_w, row_h - extra))

        label_font = QFont(self.font())
        label_font.setPointSize(8)
        label_font.setBold(True)
        error_font = QFont(self.font())
        error_font.setPointSize(7)

        for row_index, row in enumerate(rows):
            top = inner.top() + row_index * (row_h + ROW_GAP)
            colour = QColor(row["colour"])

            painter.setFont(label_font)
            painter.setPen(QPen(colour))
            metrics = painter.fontMetrics()
            painter.drawText(
                QRectF(inner.left(), top, label_width - 6, side),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(
                    row["label"], Qt.TextElideMode.ElideRight, int(label_width - 8)
                ),
            )

            for column, pixmap in enumerate(row["pixmaps"]):
                x = inner.left() + label_width + column * (cell_w + CELL_GAP)
                x += (cell_w - side) / 2
                scaled = pixmap.scaled(
                    int(side),
                    int(side),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                painter.drawPixmap(int(x), int(top), scaled)

                painter.setPen(QPen(colour, 1.4))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(QRectF(x, top, side, side), 4, 4)

                errors = row["errors"]
                if errors is None or column >= len(errors):
                    continue
                painter.setFont(error_font)
                painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
                painter.drawText(
                    QRectF(
                        inner.left() + label_width + column * (cell_w + CELL_GAP),
                        top + side + 1,
                        cell_w,
                        ERROR_HEIGHT,
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{float(errors[column]):.4f}",
                )
