"""The transformer workspace: four stages, its own state, its own workers.

It shares nothing with the other workspaces but the theme, the small widgets
and the learning-curve chart. Its reason to exist is the attention map, and its
discipline is refusing to let that map be read as an explanation until it has
been tested as one.
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

from ..core import transformer as tr
from ..core.transformer_trainer import (
    TransformerRequest,
    format_position_probe,
    format_run,
    verdict,
)
from .attention_map import AttentionMap
from .plots import LearningCurves
from .transformer.architecture_panel import TransformerArchitecturePanel
from .transformer.data_panel import TransformerDataPanel
from .transformer.inspect_panel import AVERAGE, TransformerInspectPanel
from .transformer.training_panel import TransformerTrainingPanel
from .transformer_canvas import TransformerCanvas
from .widgets import Card
from .workers import DatasetWorker, PositionProbeWorker, TransformerWorker

DATA_TAB, ARCHITECTURE_TAB, TRAINING_TAB, INSPECT_TAB = 0, 1, 2, 3

VERDICT_HEADLINES = {
    "faithful": "Faithful: this map earned its reading",
    "decorative": "Decoration: the map points, the answer does not follow",
    "distributed": "No single token matters, and the flat map says so",
    "moot": "No conclusion: the model has not learned anything yet",
}


class TransformerWorkspace(QWidget):
    """Generate a token task, train a tiny transformer, question its attention."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle: tr.TokenBundle | None = None
        self._arms: list = []
        self._data_worker: DatasetWorker | None = None
        self._train_worker = None
        self._histories: dict = {}
        self._total_epochs = 0
        self._baseline = float("nan")

        self.data_panel = TransformerDataPanel()
        self.architecture_panel = TransformerArchitecturePanel()
        self.training_panel = TransformerTrainingPanel()
        self.inspect_panel = TransformerInspectPanel()

        self.canvas = TransformerCanvas()
        self.attention = AttentionMap()
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
        tabs.addTab(self.inspect_panel, "4. Inspect")
        tabs.setMinimumWidth(400)
        self.tabs = tabs

        self.diagram_card = Card("Model shape")
        self.diagram_card.add(self.canvas, 1)
        self.attention_card = Card("Attention map")
        self.attention_card.add(self.attention, 1)
        curves_card = Card("Learning curves")
        curves_card.add(self.curves, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.diagram_card)
        right.addWidget(self.attention_card)
        right.addWidget(curves_card)
        for index, factor in ((0, 2), (1, 5), (2, 3)):
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
        self.training_panel.train_requested.connect(self._train)
        self.training_panel.probe_requested.connect(self._probe)
        self.training_panel.stop_requested.connect(self._stop)
        self.inspect_panel.show_requested.connect(self._show_map)

    # -------------------------------------------------------------------- data

    def _sync_plan(self) -> None:
        if self._bundle is not None:
            b = self._bundle
            self.architecture_panel.set_data_shape(b.length, b.vocab, b.n_outputs)
            return
        vocab = self.data_panel.planned_vocab()
        self.architecture_panel.set_data_shape(
            self.data_panel.planned_length(), vocab, vocab - 1)

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
        self.training_panel.append_log(tr.TASK_NOTES.get(bundle.task, ""))
        self.training_panel.reset_metrics()
        floor = tr.baseline(bundle)
        self._baseline = float(floor["value"])
        self.training_panel.set_baseline(self._baseline, floor["explain"])
        self.architecture_panel.set_baseline(self._baseline)
        self.curves.clear()
        self._sync_plan()
        self.status.emit(f"{bundle.name} ready")
        self.tabs.setCurrentIndex(ARCHITECTURE_TAB)

    def _on_dataset_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nDATASET FAILED\n{message}")
        QMessageBox.critical(self, "Could not generate the dataset",
                             message.split("\n\n")[0])

    def _cleanup_data_worker(self) -> None:
        if self._data_worker:
            self._data_worker.deleteLater()
        self._data_worker = None
        self.data_panel.set_busy(False)

    # ------------------------------------------------------------------ config

    def _on_config_changed(self) -> None:
        panel = self.architecture_panel
        panel.refresh_summary()
        config = panel.current_config()
        note = ""
        if not config.positional:
            note = ("Positional encoding is OFF. Every attention block after the "
                    "embedding sees an unordered set of tokens: 'first', 'after' "
                    "and 'next to' no longer exist for this model.")
        self.canvas.set_architecture(
            config.stack(), config.positional,
            f"{config.length} tokens in, {config.n_outputs} possible answers out",
            note,
        )
        self.diagram_card.set_title(f"Model shape - {config.describe()}")

    # ---------------------------------------------------------------- training

    def _request(self) -> TransformerRequest | None:
        if self._bundle is None:
            QMessageBox.information(self, "No data yet",
                                    "Generate a token dataset in the Data tab first.")
            self.tabs.setCurrentIndex(DATA_TAB)
            return None
        problems = self.architecture_panel.problems()
        if problems:
            QMessageBox.information(self, "The model cannot be built",
                                    "\n\n".join(problems))
            self.tabs.setCurrentIndex(ARCHITECTURE_TAB)
            return None
        return TransformerRequest(
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
        self._start(TransformerWorker(request, parent=self), request.epochs)

    def _probe(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        answer = QMessageBox.question(
            self, "2 training runs",
            "This trains the same transformer twice from the same initial "
            "weights: once with positional encoding, once without.\n\n"
            f"That is {request.epochs} epochs x 2.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start(PositionProbeWorker(request, parent=self), request.epochs * 2)

    def _start(self, worker, total_epochs: int) -> None:
        self._train_worker = worker
        self._total_epochs = total_epochs
        self._histories = {}
        self._arms = []

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_running(True)
        self.data_panel.set_busy(True)
        self.architecture_panel.set_running(True)
        self.inspect_panel.set_ready(False)
        self.attention.clear()
        self.curves.clear()
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit("Training...")

        worker.epoch_done.connect(self._on_epoch)
        worker.message.connect(self.training_panel.append_log)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._cleanup_train_worker)
        if isinstance(worker, PositionProbeWorker):
            worker.arm_done.connect(self._on_arm_done)
            worker.succeeded.connect(self._on_probe_done)
        else:
            worker.succeeded.connect(self._on_train_done)
        worker.start()

    def _stop(self) -> None:
        if self._train_worker:
            self._train_worker.stop()
            self.status.emit("Stopping after the current batch...")

    def _on_epoch(self, epoch: int, logs: dict, label: str) -> None:
        history = self._histories.setdefault(label or "run", {})
        for key, value in logs.items():
            history.setdefault(key, []).append(float(value))
        done = sum(len(h.get("loss", [])) for h in self._histories.values())
        self.training_panel.show_progress(done, self._total_epochs, epoch, logs, label)
        if len(self._histories) == 1 and not label:
            self.curves.update_history(history)
        else:
            self.curves.show_arms(self._histories, metric="accuracy")

    def _on_arm_done(self, label: str, outcome: dict) -> None:
        self._arms.append(outcome)
        self.training_panel.append_log("\n" + format_run(outcome))

    def _on_train_done(self, result: dict) -> None:
        self._arms = [result]
        self.training_panel.append_log("\n" + format_run(result))
        self.curves.update_history(result.get("history", {}))
        self._finish_run()

    def _on_probe_done(self, outcome: dict) -> None:
        self._arms = list(outcome.get("arms") or [])
        self.training_panel.append_log("\n" + format_position_probe(outcome))
        self._finish_run()

    def _finish_run(self) -> None:
        if not self._arms or self._bundle is None:
            return
        config = self.architecture_panel.current_config()
        self.inspect_panel.set_ready(
            True, len(self._bundle.x_val), config.n_blocks, config.n_heads,
            arms=tuple(arm["label"] for arm in self._arms),
        )
        self._show_map(0, 0, config.n_blocks - 1, AVERAGE)
        self.status.emit("Done - the attention map is on the Inspect tab")

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

    def _expected_position(self, sequence: np.ndarray) -> int:
        """Where the task says the answer lives - the green column."""
        task = self._bundle.task
        if task == tr.FIRST:
            return 0
        if task == tr.MATCH:
            where = np.where(sequence == tr.CUE)[0]
            return int(where[0]) + 1 if len(where) else -1
        return -1          # majority: nowhere in particular, by design

    def _show_map(self, arm: int, index: int, block: int, head: int) -> None:
        if not self._arms or self._bundle is None:
            return
        outcome = self._arms[min(max(0, arm), len(self._arms) - 1)]
        sequence = self._bundle.x_val[index:index + 1]
        maps = tr.attention_maps(outcome["attention_model"], sequence)
        block = min(max(0, block), len(maps) - 1)
        grid = maps[block][0]                             # (heads, L, L)
        weights = grid.mean(axis=0) if head == AVERAGE else grid[head]

        tokens = ["cue" if t == tr.CUE else str(int(t)) for t in sequence[0]]
        who = "all heads" if head == AVERAGE else f"head {head + 1}"
        self.attention.set_map(
            weights, tokens,
            f"{outcome['label']} - sequence {index}, block {block + 1}, {who}",
            expected=self._expected_position(sequence[0]),
        )
        self.attention_card.set_title(f"Attention map - {outcome['label']}")

        guess = int(tr.predict_classes(outcome["model"], sequence)[0]) + 1
        truth = int(np.argmax(self._bundle.y_val[index])) + 1
        self.inspect_panel.show_answer(
            f"The model answered token {guess}; the right answer is token {truth}. "
            f"It weighed column {self.attention.focus} most."
        )

        kind = verdict(outcome)
        faith = outcome["faithfulness"]
        self.inspect_panel.show_verdict(
            VERDICT_HEADLINES[kind],
            f"Erasing the most-attended token changed accuracy by "
            f"{-faith['attended_drop']:+.3f}; erasing a random other token, by "
            f"{-faith['control_drop']:+.3f}. Measured on block "
            f"{faith['block'] + 1} over {faith['n']} validation sequences.",
        )

    # ------------------------------------------------------------------- state

    def _forget_model(self) -> None:
        self._arms = []
        self._histories = {}
        self.inspect_panel.set_ready(False)
        self.attention.clear()

    def shutdown(self) -> None:
        for worker in (self._train_worker, self._data_worker):
            if worker:
                worker.stop()
                worker.wait(4000)
