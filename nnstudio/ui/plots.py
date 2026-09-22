"""Learning curves rendered with matplotlib inside the Qt window."""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import theme

METRIC_TITLES = {
    "accuracy": "Accuracy",
    "mae": "Mean absolute error",
    "mse": "Mean squared error",
}


class LearningCurves(FigureCanvasQTAgg):
    """Two stacked axes: loss on top, the task metric below."""

    def __init__(self, parent=None):
        figure = Figure(figsize=(5, 3.2), dpi=100, facecolor=theme.SURFACE_ALT)
        super().__init__(figure)
        self.setParent(parent)
        self._loss_ax = figure.add_subplot(211)
        self._metric_ax = figure.add_subplot(212)
        figure.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.17, hspace=0.62)
        self.setMinimumHeight(260)
        self.clear()

    def _style(self, axes, title: str) -> None:
        axes.set_facecolor(theme.SURFACE_ALT)
        axes.set_title(title, color=theme.TEXT, fontsize=10, loc="left", pad=6)
        axes.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        axes.grid(True, color=theme.BORDER, linewidth=0.6, alpha=0.6)
        for spine in axes.spines.values():
            spine.set_color(theme.BORDER)
        axes.set_xlabel("epoch", color=theme.TEXT_MUTED, fontsize=8)

    def clear(self) -> None:
        for axes, title in ((self._loss_ax, "Loss"), (self._metric_ax, "Metric")):
            axes.clear()
            self._style(axes, title)
            axes.text(
                0.5,
                0.5,
                "waiting for training",
                transform=axes.transAxes,
                ha="center",
                va="center",
                color=theme.TEXT_MUTED,
                fontsize=9,
            )
        self.draw_idle()

    def update_history(self, history: dict) -> None:
        """Redraw from a Keras-style history dict."""
        if not history or "loss" not in history:
            self.clear()
            return

        epochs = range(1, len(history["loss"]) + 1)

        self._loss_ax.clear()
        self._style(self._loss_ax, "Loss")
        self._loss_ax.plot(
            epochs, history["loss"], color=theme.ACCENT, linewidth=1.8, label="train"
        )
        if "val_loss" in history:
            self._loss_ax.plot(
                epochs,
                history["val_loss"],
                color=theme.OUTPUT_COLOR,
                linewidth=1.8,
                linestyle="--",
                label="validation",
            )
        self._legend(self._loss_ax)

        metric_key = next(
            (k for k in ("accuracy", "mae", "mse") if k in history), None
        )
        self._metric_ax.clear()
        title = METRIC_TITLES.get(metric_key, "Metric")
        self._style(self._metric_ax, title)
        if metric_key:
            self._metric_ax.plot(
                epochs,
                history[metric_key],
                color=theme.SUCCESS,
                linewidth=1.8,
                label="train",
            )
            val_key = f"val_{metric_key}"
            if val_key in history:
                self._metric_ax.plot(
                    epochs,
                    history[val_key],
                    color=theme.INPUT_COLOR,
                    linewidth=1.8,
                    linestyle="--",
                    label="validation",
                )
            self._legend(self._metric_ax)

        self.draw_idle()

    def show_arms(self, histories: dict) -> None:
        """Overlay several named runs, e.g. the residual and plain arms."""
        if not histories:
            self.clear()
            return

        palette = {
            "residual": theme.SUCCESS,
            "plain": theme.DANGER,
        }
        spare = (theme.ACCENT, theme.OUTPUT_COLOR, theme.INPUT_COLOR)

        self._loss_ax.clear()
        self._style(self._loss_ax, "Validation loss")
        self._metric_ax.clear()
        self._style(self._metric_ax, "Validation accuracy")

        for index, (label, history) in enumerate(histories.items()):
            colour = palette.get(label, spare[index % len(spare)])
            losses = history.get("val_loss") or history.get("loss") or []
            if losses:
                self._loss_ax.plot(
                    range(1, len(losses) + 1), losses,
                    color=colour, linewidth=1.8, label=label,
                )
            accuracy = history.get("val_accuracy") or history.get("accuracy") or []
            if accuracy:
                self._metric_ax.plot(
                    range(1, len(accuracy) + 1), accuracy,
                    color=colour, linewidth=1.8, label=label,
                )

        for axes in (self._loss_ax, self._metric_ax):
            if axes.has_data():
                self._legend(axes)
        self.draw_idle()

    def show_folds(self, result: dict) -> None:
        """Per-fold bars with the mean line and the +/- one std band.

        The spread is the point of the chart: a tall band means the single
        holdout number you were reading was mostly luck.
        """
        folds = result.get("folds") or []
        metric = result.get("metric")
        if not folds or not metric:
            self.clear()
            return

        positions = [entry["fold"] for entry in folds]
        stats = result["summary"][metric]
        values = stats["values"]
        mean, std = stats["mean"], stats["std"]

        self._loss_ax.clear()
        # The stats live in the title: a legend box would sit on top of a bar.
        title = (
            f"{METRIC_TITLES.get(metric, metric)} per fold  -  "
            f"mean {mean:.4f}  +/- {std:.4f}"
        )
        self._style(self._loss_ax, title)
        self._loss_ax.set_xlabel("fold", color=theme.TEXT_MUTED, fontsize=8)
        self._loss_ax.bar(
            positions, values, color=theme.ACCENT, width=0.55, zorder=3
        )
        self._loss_ax.axhspan(
            mean - std, mean + std, color=theme.OUTPUT_COLOR, alpha=0.18, zorder=2
        )
        self._loss_ax.axhline(mean, color=theme.SUCCESS, linewidth=1.6, zorder=4)
        self._loss_ax.set_xticks(positions)
        span = max(std * 3, (max(values) - min(values)) * 1.4, 1e-6)
        self._loss_ax.set_ylim(min(values) - span * 0.4, max(values) + span * 0.4)

        self._metric_ax.clear()
        self._style(self._metric_ax, "Validation loss per fold")
        self._metric_ax.set_xlabel("fold", color=theme.TEXT_MUTED, fontsize=8)
        losses = [entry["scores"].get("loss", float("nan")) for entry in folds]
        self._metric_ax.bar(
            positions, losses, color=theme.HIDDEN_COLOR, width=0.55, zorder=3
        )
        self._metric_ax.set_xticks(positions)
        self.draw_idle()

    @staticmethod
    def _legend(axes) -> None:
        legend = axes.legend(
            loc="best",
            fontsize=8,
            facecolor=theme.ELEVATED,
            edgecolor=theme.BORDER,
            framealpha=0.9,
        )
        if legend:
            for text in legend.get_texts():
                text.set_color(theme.TEXT_MUTED)
