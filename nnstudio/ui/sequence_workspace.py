"""The sequence workspace: four stages, its own state, its own workers.

It shares nothing with the other three except the theme, the small widgets and
the learning-curve chart. That is on purpose. A recurrent network is not a later
stage of a dense one, and the question it exists to answer - does order carry
anything here - has no counterpart in the other workspaces.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import sequences as sq
from ..core.sequence_trainer import (
    SWEEP_KINDS,
    SequenceRequest,
    format_order_probe,
    format_run,
    format_sweep,
)
from .plots import LearningCurves
from .sequence_canvas import SequenceCanvas
from .sequences.architecture_panel import SequenceArchitecturePanel
from .sequences.data_panel import SequenceDataPanel
from .sequences.predict_panel import SequencePredictPanel
from .sequences.training_panel import SequenceTrainingPanel
from .series_plot import SeriesPlot
from .widgets import Card
from .workers import (
    ArchSweepWorker,
    DatasetWorker,
    OrderProbeWorker,
    SequenceWorker,
)

DATA_TAB, ARCHITECTURE_TAB, TRAINING_TAB, PREDICT_TAB = 0, 1, 2, 3


class SequenceWorkspace(QWidget):
    """Generate a sequence task, choose a model, train it, question the result."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle: sq.SequenceBundle | None = None
        self._model = None
        self._data_worker: DatasetWorker | None = None
        self._train_worker = None
        self._histories: dict = {}
        self._total_epochs = 0
        self._baseline = float("nan")
        self._lower_is_better = True
        self._arms: list = []

        self.data_panel = SequenceDataPanel()
        self.architecture_panel = SequenceArchitecturePanel()
        self.training_panel = SequenceTrainingPanel()
        self.predict_panel = SequencePredictPanel()

        self.canvas = SequenceCanvas()
        self.series = SeriesPlot()
        self.curves = LearningCurves()

        self._build_layout()
        self._connect()
        self._sync_plan()
        self._on_config_changed()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self.data_panel, "1. Data")
        tabs.addTab(self.architecture_panel, "2. Architecture")
        tabs.addTab(self.training_panel, "3. Training")
        tabs.addTab(self.predict_panel, "4. Inspect")
        tabs.setMinimumWidth(400)
        self.tabs = tabs

        self.diagram_card = Card("Model shape")
        self.diagram_card.add(self.canvas, 1)
        self.series_card = Card("The data and the answers")
        self.series_card.add(self.series, 1)
        curves_card = Card("Learning curves")
        curves_card.add(self.curves, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.diagram_card)
        right.addWidget(self.series_card)
        right.addWidget(curves_card)
        for index, factor in ((0, 3), (1, 4), (2, 3)):
            right.setStretchFactor(index, factor)
        right.setChildrenCollapsible(False)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(tabs)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 1000])
        split.setChildrenCollapsible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

    def _connect(self) -> None:
        self.data_panel.build_requested.connect(self._build_dataset)
        self.data_panel.plan_changed.connect(self._sync_plan)

        self.architecture_panel.config_changed.connect(self._on_config_changed)
        self.architecture_panel.preset_requested.connect(self._apply_preset)

        self.training_panel.train_requested.connect(self._train)
        self.training_panel.sweep_requested.connect(self._sweep)
        self.training_panel.probe_requested.connect(self._probe)
        self.training_panel.stop_requested.connect(self._stop)

        self.predict_panel.forecast_requested.connect(self._show_forecast)
        self.predict_panel.confusion_requested.connect(self._show_confusion)
        self.predict_panel.windows_requested.connect(self._show_windows)

    # -------------------------------------------------------------------- data

    def _sync_plan(self) -> None:
        if self._bundle is not None:
            self.architecture_panel.set_data_shape(
                self._bundle.length, self._bundle.kind, self._bundle.n_outputs
            )
            return
        task = self.data_panel.current_task()
        kind = sq.TASK_KINDS[task]
        self.architecture_panel.set_data_shape(
            self.data_panel.planned_length(), kind, 2 if kind == sq.CLASSIFY else 1
        )

    def _build_dataset(self, builder, description: str) -> None:
        if self._data_worker or self._train_worker:
            return
        self.data_panel.set_busy(True)
        self.training_panel.append_log(description)
        self.status.emit(description)

        self._data_worker = DatasetWorker(builder, description, parent=self)
        self._data_worker.message.connect(self.status.emit)
        self._data_worker.progress.connect(self.data_panel.show_progress)
        self._data_worker.succeeded.connect(self._on_dataset_ready)
        self._data_worker.failed.connect(self._on_dataset_failed)
        self._data_worker.finished.connect(self._cleanup_data_worker)
        self._data_worker.start()

    def _on_dataset_ready(self, bundle) -> None:
        self._bundle = bundle
        self._forget_model()

        self.data_panel.show_summary(bundle.describe())
        self.training_panel.append_log(bundle.describe())
        self.training_panel.append_log(sq.TASK_NOTES.get(bundle.task, ""))
        self.training_panel.reset_metrics()
        self._publish_baseline(sq.baseline(bundle))
        self.curves.clear()

        self._sync_plan()
        self.predict_panel.set_has_data(True)
        self._show_windows()
        self.status.emit(f"{bundle.name} ready")
        self.tabs.setCurrentIndex(ARCHITECTURE_TAB)

    def _on_dataset_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nDATASET FAILED\n{message}")
        self.status.emit("Could not generate the dataset")
        QMessageBox.critical(self, "Could not generate the dataset",
                             message.split("\n\n")[0])

    def _cleanup_data_worker(self) -> None:
        if self._data_worker:
            self._data_worker.deleteLater()
        self._data_worker = None
        self.data_panel.set_busy(False)

    def _publish_baseline(self, floor: dict) -> None:
        """One place computes the floor, and both tabs show the same number."""
        self._baseline = float(floor["value"])
        self._lower_is_better = bool(floor["better_is_lower"])
        self.training_panel.set_baseline(
            self._baseline, floor["metric"], self._lower_is_better, floor["explain"]
        )
        self.architecture_panel.set_baseline(
            self._baseline, floor["metric"], self._lower_is_better
        )

    # ------------------------------------------------------------------ config

    def _on_config_changed(self) -> None:
        panel = self.architecture_panel
        panel.refresh_summary()
        config = panel.current_config()

        caption = (
            f"{config.length} steps in, "
            + ("one number out" if config.task_kind == sq.FORECAST
               else f"{config.n_outputs} classes out")
        )
        note = ""
        if not config.reads_order:
            note = (
                f"{config.kind} has no state carried from one step to the next, "
                "so there is no loop in this picture. That does not make it "
                "order-blind on a fixed window - it just has to learn every "
                "position separately."
            )
        self.canvas.set_architecture(config.stack(), config.length,
                                     config.reads_order, caption, note)
        self.diagram_card.set_title(f"Model shape - {config.describe()}")

    def _apply_preset(self, epochs: int, batch: int) -> None:
        self.training_panel.apply_preset(epochs, batch)
        names = ", ".join(SWEEP_KINDS)
        self.training_panel.append_log(
            f"Sweep preset applied: {epochs} epochs per arm, batch {batch}.\n"
            f"Press 'Compare architectures' to train {names} from the same "
            "initial weights on the same windows."
        )
        self.tabs.setCurrentIndex(TRAINING_TAB)

    # ---------------------------------------------------------------- training

    def _request(self) -> SequenceRequest | None:
        if self._bundle is None:
            QMessageBox.information(self, "No data yet",
                                    "Generate a sequence dataset in the Data tab first.")
            self.tabs.setCurrentIndex(DATA_TAB)
            return None

        problems = self.architecture_panel.problems()
        if problems:
            QMessageBox.information(self, "The model cannot be built",
                                    "\n\n".join(problems))
            self.tabs.setCurrentIndex(ARCHITECTURE_TAB)
            return None

        return SequenceRequest(
            data=self._bundle,
            config=self.architecture_panel.current_config(),
            epochs=self.training_panel.epochs_spin.value(),
            batch_size=self.training_panel.batch_spin.value(),
            early_stopping=self.training_panel.early_check.isChecked(),
        )

    def _train(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        self._start(SequenceWorker(request, parent=self), request.epochs, single=True)

    def _sweep(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        answer = QMessageBox.question(
            self,
            f"{len(SWEEP_KINDS)} training runs",
            f"This trains the same task {len(SWEEP_KINDS)} times - "
            f"{', '.join(SWEEP_KINDS)} - from identical initial weights.\n\n"
            f"That is {request.epochs} epochs x {len(SWEEP_KINDS)}.\n\n"
            "Everything else stays fixed, so the architecture is the only "
            "variable.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start(ArchSweepWorker(request, SWEEP_KINDS, parent=self),
                    request.epochs * len(SWEEP_KINDS), single=False)

    def _probe(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        answer = QMessageBox.question(
            self,
            "2 training runs",
            "This trains the same model twice: once on the real data, once "
            "with every sequence's timesteps permuted.\n\n"
            f"That is {request.epochs} epochs x 2.\n\n"
            "If the two scores match, order carried nothing and a recurrent "
            "layer is wasted on this task.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start(OrderProbeWorker(request, parent=self),
                    request.epochs * 2, single=False)

    def _start(self, worker, total_epochs: int, single: bool) -> None:
        self._train_worker = worker
        self._total_epochs = total_epochs
        self._histories = {}
        self._arms = []
        self._model = None

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_running(True)
        self.data_panel.set_busy(True)
        self.architecture_panel.set_running(True)
        self.predict_panel.set_ready(False)
        self.curves.clear()
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit("Training...")

        worker.epoch_done.connect(self._on_epoch)
        worker.message.connect(self._on_message)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._cleanup_train_worker)
        if single:
            worker.succeeded.connect(self._on_train_done)
        elif isinstance(worker, OrderProbeWorker):
            worker.arm_done.connect(self._on_arm_done)
            worker.succeeded.connect(self._on_probe_done)
        else:
            worker.arm_done.connect(self._on_arm_done)
            worker.succeeded.connect(self._on_sweep_done)
        worker.start()

    def _stop(self) -> None:
        if self._train_worker:
            self._train_worker.stop()
            self.status.emit("Stopping after the current batch...")

    def _on_message(self, text: str) -> None:
        self.training_panel.append_log(text)
        self.status.emit(text.strip().splitlines()[-1][:110] if text.strip() else "")

    def _on_epoch(self, epoch: int, logs: dict, label: str) -> None:
        history = self._histories.setdefault(label or "run", {})
        for key, value in logs.items():
            history.setdefault(key, []).append(float(value))

        done = sum(len(h.get("loss", [])) for h in self._histories.values())
        self.training_panel.show_progress(done, self._total_epochs, epoch, logs, label)
        if len(self._histories) == 1 and not label:
            self.curves.update_history(history)
        else:
            self.curves.show_arms(
                self._histories,
                metric="accuracy" if not self._lower_is_better else "mae",
                baseline=self._baseline if self._lower_is_better else None,
            )

    def _on_arm_done(self, label: str, outcome: dict) -> None:
        self._arms.append(outcome)
        self.training_panel.append_log(f"\n{format_run(outcome)}")

    def _on_train_done(self, result: dict) -> None:
        self._model = result.get("model")
        self.training_panel.append_log("\n" + format_run(result))
        self.curves.update_history(result.get("history", {}))
        self._finish_run()

    def _on_sweep_done(self, outcome: dict) -> None:
        arms = outcome.get("arms") or []
        self._model = arms[-1]["model"] if arms else None
        self.training_panel.append_log("\n" + format_sweep(outcome))
        self._finish_run()

    def _on_probe_done(self, outcome: dict) -> None:
        arms = outcome.get("arms") or []
        # The real-order arm is the one worth keeping for the Inspect tab; the
        # shuffled one exists only to be compared against.
        self._model = arms[0]["model"] if arms else None
        self.training_panel.append_log("\n" + format_order_probe(outcome))
        self._finish_run()

    def _finish_run(self) -> None:
        ready = self._model is not None and self._bundle is not None
        self.predict_panel.set_ready(ready, self._bundle.kind if ready else "")
        if ready:
            if self._bundle.kind == sq.FORECAST:
                self._show_forecast(int(self.predict_panel.count_spin.value()))
            else:
                self._show_confusion()
        self.status.emit("Done")

    def _on_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nTRAINING FAILED\n{message}")
        self.status.emit("Training failed")
        QMessageBox.critical(self, "Training failed", message.split("\n\n")[0])

    def _cleanup_train_worker(self) -> None:
        if self._train_worker:
            self._train_worker.deleteLater()
        self._train_worker = None
        self.training_panel.set_running(False)
        self.data_panel.set_busy(False)
        self.architecture_panel.set_running(False)

    # ----------------------------------------------------------------- inspect

    def _show_windows(self) -> None:
        if self._bundle is None:
            return
        bundle = self._bundle
        labels = None
        if bundle.kind == sq.CLASSIFY:
            indices = np.argmax(bundle.y_train[:5], axis=1)
            labels = [bundle.class_names[i] for i in indices]
        self.series.show_windows(
            bundle.x_train, labels,
            f"Five windows from {bundle.name}"
            + (" - TIMESTEPS SHUFFLED" if bundle.shuffled else ""),
        )
        self.series_card.set_title("The data and the answers - raw windows")

    def _show_forecast(self, count: int) -> None:
        if self._model is None or self._bundle is None:
            return
        rows = sq.forecast_rows(self._model, self._bundle, count)
        model_mse = float(np.mean((rows["actual"] - rows["predicted"]) ** 2))
        floor_mse = float(np.mean((rows["actual"] - rows["persistence"]) ** 2))
        self.series.show_forecast(
            rows,
            f"First {len(rows['actual'])} validation steps - model {model_mse:.5f}, "
            f"persistence {floor_mse:.5f}",
        )
        self.series_card.set_title("The data and the answers - forecast")
        ratio = floor_mse / model_mse if model_mse > 0 else float("inf")
        verdict = (f"{ratio:.2f}x better" if ratio >= 1.0
                   else f"{1 / ratio:.2f}x WORSE")
        self.predict_panel.show_result(
            f"Model {model_mse:.5f} vs persistence {floor_mse:.5f}  ({verdict})",
            "Both lines are on the chart. If they sit on top of each other, the "
            "model is repeating the last value and has learned nothing, however "
            "well the curve appears to follow the truth.",
        )

    def _show_confusion(self) -> None:
        if self._model is None or self._bundle is None:
            return
        table = sq.confusion(self._model, self._bundle)
        correct = int(np.trace(table))
        total = int(table.sum())
        accuracy = correct / max(total, 1)
        self.series.show_confusion(
            table, self._bundle.class_names,
            f"{correct} of {total} correct ({accuracy:.1%})",
        )
        self.series_card.set_title("The data and the answers - confusion")
        self.predict_panel.show_result(
            f"{accuracy:.1%} correct against a {self._baseline:.1%} majority class",
            "The diagonal is what it got right. An off-diagonal cell that is "
            "much larger than its mirror means the model prefers one answer - "
            "which is worth knowing before you trust the accuracy.",
        )

    # ------------------------------------------------------------------- state

    def _forget_model(self) -> None:
        self._model = None
        self._histories = {}
        self._arms = []
        self.predict_panel.set_ready(False)

    def shutdown(self) -> None:
        for worker in (self._train_worker, self._data_worker):
            if worker:
                worker.stop()
                worker.wait(4000)
