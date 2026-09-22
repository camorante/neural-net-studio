"""The forecast chart, the confusion table, and a look at the raw windows.

The forecast chart always draws three lines, never one. A prediction plotted on
its own against the truth is the most flattering picture in this whole subject:
a curve that simply repeats the previous value tracks the target beautifully by
eye and is worth exactly nothing. Persistence goes on the same axes so that
flattery is impossible - if the model's line sits on top of the persistence
line, the student can see for themselves that no work was done.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import theme


class SeriesPlot(FigureCanvasQTAgg):
    """One axis that shows whichever view the current task calls for."""

    def __init__(self, parent=None):
        figure = Figure(figsize=(5, 3.0), dpi=100, facecolor=theme.SURFACE_ALT)
        super().__init__(figure)
        self.setParent(parent)
        self._ax = figure.add_subplot(111)
        figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
        self.setMinimumHeight(240)
        self._caption = ""
        self.clear()

    @property
    def caption(self) -> str:
        return self._caption

    def _style(self, title: str) -> None:
        self._ax.set_facecolor(theme.SURFACE_ALT)
        self._ax.set_title(title, color=theme.TEXT, fontsize=10, loc="left", pad=6)
        self._ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        self._ax.grid(True, color=theme.BORDER, linewidth=0.6, alpha=0.6)
        for spine in self._ax.spines.values():
            spine.set_color(theme.BORDER)

    def clear(self) -> None:
        self._caption = ""
        self._ax.clear()
        self._style("Nothing to show yet")
        self._ax.text(0.5, 0.5, "build a dataset to begin",
                      transform=self._ax.transAxes, ha="center", va="center",
                      color=theme.TEXT_MUTED, fontsize=9)
        self.draw_idle()

    # ------------------------------------------------------------------ views

    def show_windows(self, x: np.ndarray, labels=None, caption: str = "") -> None:
        """A handful of raw windows, so the data is looked at before it is fed in."""
        self._caption = caption
        self._ax.clear()
        self._style(caption or "A few windows from the training set")
        self._ax.set_xlabel("step in the window", color=theme.TEXT_MUTED, fontsize=8)
        colours = (theme.ACCENT, theme.SUCCESS, theme.OUTPUT_COLOR,
                   theme.HIDDEN_COLOR, theme.INPUT_COLOR)
        for index, window in enumerate(x[:5]):
            name = labels[index] if labels is not None and index < len(labels) else None
            self._ax.plot(window[:, 0], color=colours[index % len(colours)],
                          linewidth=1.5, marker="o", markersize=2.6, label=name)
        if labels is not None:
            self._legend()
        self.draw_idle()

    def show_forecast(self, rows: dict, caption: str = "") -> None:
        """Actual, the model, and persistence - all three, always."""
        self._caption = caption
        self._ax.clear()
        self._style(caption or "Validation forecast")
        self._ax.set_xlabel("validation step", color=theme.TEXT_MUTED, fontsize=8)

        actual = np.asarray(rows.get("actual", []))
        steps = range(1, len(actual) + 1)
        self._ax.plot(steps, actual, color=theme.TEXT, linewidth=1.8, label="actual")
        self._ax.plot(steps, rows.get("persistence", []), color=theme.TEXT_MUTED,
                      linewidth=1.2, linestyle=":", label="persistence")
        self._ax.plot(steps, rows.get("predicted", []), color=theme.ACCENT,
                      linewidth=1.6, label="model")
        self._legend()
        self.draw_idle()

    def show_confusion(self, table: np.ndarray, class_names, caption: str = "") -> None:
        """Rows are the truth, columns are the answer, counts written in."""
        self._caption = caption
        self._ax.clear()
        self._style(caption or "Confusion on the validation set")
        table = np.asarray(table)
        self._ax.imshow(table, cmap="Blues", aspect="auto")
        self._ax.set_xticks(range(len(class_names)))
        self._ax.set_yticks(range(len(class_names)))
        self._ax.set_xticklabels(class_names, fontsize=8)
        self._ax.set_yticklabels(class_names, fontsize=8)
        self._ax.set_xlabel("answered", color=theme.TEXT_MUTED, fontsize=8)
        self._ax.set_ylabel("actually", color=theme.TEXT_MUTED, fontsize=8)
        self._ax.grid(False)
        highest = table.max() if table.size else 1
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                value = int(table[row, column])
                self._ax.text(
                    column, row, str(value), ha="center", va="center", fontsize=10,
                    color=theme.BACKGROUND if value > highest * 0.55 else theme.TEXT,
                )
        self.draw_idle()

    def _legend(self) -> None:
        legend = self._ax.legend(loc="best", fontsize=8, facecolor=theme.ELEVATED,
                                 edgecolor=theme.BORDER, framealpha=0.9)
        if legend:
            for text in legend.get_texts():
                text.set_color(theme.TEXT_MUTED)
