"""K-fold cross-validation: an estimate of an architecture, not a model.

The k models trained here are deliberately thrown away. What survives is the
mean and the spread of their scores, which is the only honest way to judge a
configuration when a single holdout split is too small to trust.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .dataset import TASK_REGRESSION, PreparationSpec
from .model_builder import NetworkConfig
from .trainer import TrainingRequest, TrainingRun


@dataclass
class CrossValRequest:
    """The same architecture and hyperparameters, evaluated over k folds."""

    spec: PreparationSpec
    template: NetworkConfig
    k: int = 5
    epochs: int = 60
    batch_size: int = 32
    shuffle: bool = True
    early_stopping: bool = False
    patience: int = 10


class CrossValRun:
    """Trains k models, keeps only their scores."""

    def __init__(self, request: CrossValRequest, on_fold=None, on_message=None):
        self.request = request
        self.on_fold = on_fold
        self.on_message = on_message
        self._stop = False
        self._current: TrainingRun | None = None

    def stop(self) -> None:
        self._stop = True
        if self._current:
            self._current.stop()

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def run(self) -> dict:
        request = self.request
        self._say(f"Building {request.k} folds...")
        fold_set = request.spec.folds(request.k)

        if not fold_set.stratified and fold_set.task != TASK_REGRESSION:
            self._say(
                "Note: plain K-fold was used because at least one class has fewer "
                "than k members, so class balance is not guaranteed per fold."
            )
        else:
            kind = "K-fold" if fold_set.task == TASK_REGRESSION else "Stratified K-fold"
            self._say(f"{kind} over {len(fold_set)} folds.")

        results: list = []
        for index, bundle in enumerate(fold_set, start=1):
            if self._stop:
                break
            # Each fold refits its own preprocessing, so the encoded feature
            # count can differ slightly when a rare category is absent.
            config = replace(
                request.template, n_inputs=bundle.n_inputs, n_outputs=bundle.n_outputs
            )
            self._say(
                f"Fold {index}/{len(fold_set)}: training on {bundle.n_train} rows, "
                f"validating on {bundle.n_val}."
            )
            self._current = TrainingRun(
                TrainingRequest(
                    config=config,
                    data=bundle,
                    epochs=request.epochs,
                    batch_size=request.batch_size,
                    shuffle=request.shuffle,
                    early_stopping=request.early_stopping,
                    patience=request.patience,
                )
            )
            outcome = self._current.run()
            self._current = None

            scores = outcome.get("final_scores", {})
            entry = {
                "fold": index,
                "scores": scores,
                "n_train": bundle.n_train,
                "n_val": bundle.n_val,
                "epochs_run": outcome["epochs_run"],
            }
            results.append(entry)
            if self.on_fold:
                self.on_fold(index, len(fold_set), entry)

        if not results:
            return {"folds": [], "summary": {}, "metric": None, "stopped": True}

        summary = _summarise(results)
        return {
            "folds": results,
            "summary": summary,
            "metric": _primary_metric(results),
            "task": fold_set.task,
            "stratified": fold_set.stratified,
            "k": len(fold_set),
            "completed": len(results),
            "stopped": self._stop,
            "rows_validated": sum(r["n_val"] for r in results),
        }


def _primary_metric(results: list) -> str | None:
    """The score a human actually reads: accuracy, else mae, else loss."""
    keys = set()
    for entry in results:
        keys.update(entry["scores"])
    for candidate in ("accuracy", "mae", "mse", "loss"):
        if candidate in keys:
            return candidate
    return None


def _summarise(results: list) -> dict:
    """Mean, standard deviation and range per metric across the folds."""
    keys = sorted({k for entry in results for k in entry["scores"]})
    summary = {}
    for key in keys:
        values = np.array(
            [entry["scores"][key] for entry in results if key in entry["scores"]],
            dtype=float,
        )
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": [float(v) for v in values],
        }
    return summary


def format_summary(result: dict) -> str:
    """A short, honest report of a finished cross-validation."""
    if not result.get("folds"):
        return "Cross-validation produced no folds."

    metric = result.get("metric")
    lines = []
    for entry in result["folds"]:
        parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(entry["scores"].items()))
        lines.append(
            f"  fold {entry['fold']}/{result['k']}  "
            f"({entry['n_train']} train / {entry['n_val']} val)  {parts}"
        )

    if metric:
        stats = result["summary"][metric]
        lines.append("")
        lines.append(
            f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}   "
            f"(range {stats['min']:.4f} to {stats['max']:.4f})"
        )
        lines.append(
            f"  {result['rows_validated']} rows validated, each exactly once."
        )
    if result.get("stopped"):
        lines.append("  Interrupted - this is a partial estimate.")
    return "\n".join(lines)
