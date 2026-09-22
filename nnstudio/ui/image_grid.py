"""Image previews and prediction grids, drawn straight from numpy arrays."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from . import theme

COLUMNS = 6
CELL_GAP = 8
CAPTION_HEIGHT = 30


def to_qimage(frame: np.ndarray) -> QImage:
    """Turn a float32 HxWx3 array in 0..255 into an RGB888 QImage."""
    data = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
    height, width = data.shape[0], data.shape[1]
    if data.ndim == 2 or data.shape[2] == 1:
        data = np.repeat(data.reshape(height, width, 1), 3, axis=2)
        data = np.ascontiguousarray(data)
    image = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()  # detach from the numpy buffer before it is freed


class ImageGrid(QWidget):
    """A grid of images with optional predicted/true captions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cells: list = []
        self._placeholder = "Build a dataset to see sample images here"
        self.setMinimumHeight(220)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._cells = []
        self.update()

    def show_samples(self, images: np.ndarray, labels, class_names: list) -> None:
        """Images with their true class only."""
        self._cells = [
            {
                "pixmap": QPixmap.fromImage(to_qimage(frame)),
                "title": class_names[int(label)],
                "subtitle": "",
                "state": "neutral",
            }
            for frame, label in zip(images, labels)
        ]
        self.update()

    def show_predictions(
        self, images: np.ndarray, truths, probabilities: np.ndarray, class_names: list
    ) -> None:
        """Images with predicted class, confidence, and right/wrong colouring."""
        cells = []
        for frame, truth, probs in zip(images, truths, probabilities):
            predicted = int(np.argmax(probs))
            truth = int(truth)
            correct = predicted == truth
            subtitle = (
                f"{probs[predicted] * 100:.0f}% confident"
                if correct
                else f"true: {class_names[truth]}"
            )
            cells.append(
                {
                    "pixmap": QPixmap.fromImage(to_qimage(frame)),
                    "title": class_names[predicted],
                    "subtitle": subtitle,
                    "state": "correct" if correct else "wrong",
                }
            )
        self._cells = cells
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

        if not self._cells:
            painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        inner = rect.adjusted(12, 12, -12, -12)
        columns = min(COLUMNS, len(self._cells))
        rows = max(1, (len(self._cells) + columns - 1) // columns)
        cell_w = (inner.width() - CELL_GAP * (columns - 1)) / columns
        cell_h = (inner.height() - CELL_GAP * (rows - 1)) / rows
        side = max(24.0, min(cell_w, cell_h - CAPTION_HEIGHT))

        title_font = QFont(self.font())
        title_font.setPointSize(9)
        title_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(8)

        colours = {
            "neutral": QColor(theme.BORDER_STRONG),
            "correct": QColor(theme.SUCCESS),
            "wrong": QColor(theme.DANGER),
        }

        for index, cell in enumerate(self._cells):
            column, row = index % columns, index // columns
            x = inner.left() + column * (cell_w + CELL_GAP) + (cell_w - side) / 2
            y = inner.top() + row * (cell_h + CELL_GAP)

            frame_rect = QRectF(x, y, side, side)
            scaled = cell["pixmap"].scaled(
                int(side),
                int(side),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            painter.drawPixmap(int(x), int(y), scaled)

            painter.setPen(QPen(colours[cell["state"]], 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(frame_rect, 5, 5)

            caption = QRectF(
                inner.left() + column * (cell_w + CELL_GAP),
                y + side + 3,
                cell_w,
                14,
            )
            painter.setFont(title_font)
            painter.setPen(QPen(colours[cell["state"]]))
            metrics = painter.fontMetrics()
            painter.drawText(
                caption,
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(
                    cell["title"], Qt.TextElideMode.ElideRight, int(cell_w)
                ),
            )

            if cell["subtitle"]:
                painter.setFont(small_font)
                painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
                metrics = painter.fontMetrics()
                painter.drawText(
                    QRectF(caption.left(), caption.bottom(), cell_w, 13),
                    Qt.AlignmentFlag.AlignCenter,
                    metrics.elidedText(
                        cell["subtitle"], Qt.TextElideMode.ElideRight, int(cell_w)
                    ),
                )
