"""The dense workspace: the tabular four stages plus their own visuals."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.crossval import CrossValRequest, format_summary
from ..core.model_builder import NetworkConfig, validate
from ..core.trainer import TrainingRequest
from .architecture_panel import ArchitecturePanel
from .data_panel import DataPanel
from .network_canvas import NetworkCanvas
from .plots import LearningCurves
from .predict_panel import PredictPanel
from .training_panel import TrainingPanel
from .widgets import Card
from .workers import CrossValWorker, TrainingWorker

DATA_TAB, ARCHITECTURE_TAB, TRAINING_TAB, PREDICT_TAB = range(4)


class DenseWorkspace(QWidget):
    """Everything the fully connected path needs, isolated from the CNN path."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle = None
        self._config: NetworkConfig | None = None
        self._model = None
        self._worker: TrainingWorker | None = None
        self._cv_worker: CrossValWorker | None = None
        self._history: dict = {}
        self._total_epochs = 0

        self.data_panel = DataPanel()
        self.architecture_panel = ArchitecturePanel()
        self.training_panel = TrainingPanel()
        self.predict_panel = PredictPanel()

        self.canvas = NetworkCanvas()
        self.curves = LearningCurves()

        self._build_layout()
        self._connect()
        self.architecture_panel._emit()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self.data_panel, "1. Data")
        tabs.addTab(self.architecture_panel, "2. Architecture")
        tabs.addTab(self.training_panel, "3. Training")
        tabs.addTab(self.predict_panel, "4. Predict")
        tabs.setMinimumWidth(400)
        self.tabs = tabs

        self.diagram_card = Card("Network diagram")
        self.diagram_card.add(self.canvas, 1)
        curves_card = Card("Learning curves")
        curves_card.add(self.curves, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.diagram_card)
        right.addWidget(curves_card)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
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
        self.data_panel.dataset_ready.connect(self._on_dataset_ready)
        self.data_panel.status.connect(self.status.emit)
        self.architecture_panel.config_changed.connect(self._on_config_changed)
        self.training_panel.start_requested.connect(self._start_training)
        self.training_panel.stop_requested.connect(self._stop_training)
        self.training_panel.export_requested.connect(self._export_model)
        self.training_panel.crossval_requested.connect(self._start_crossval)
        self.predict_panel.predict_requested.connect(self._predict)

    # ------------------------------------------------------------------ events

    def _on_dataset_ready(self, bundle) -> None:
        self._bundle = bundle
        self._model = None
        self._history = {}
        self.architecture_panel.apply_dataset(bundle)
        self.predict_panel.set_bundle(bundle)
        self.predict_panel.set_model_ready(False)
        self.training_panel.set_export_enabled(False)
        self.training_panel.reset_metrics()
        self.curves.clear()
        self.tabs.setCurrentIndex(ARCHITECTURE_TAB)

    def _on_config_changed(self, config: NetworkConfig) -> None:
        self._config = config
        self.canvas.set_columns(config.layer_summary())
        depth = len(config.hidden)
        self.diagram_card.set_title(
            f"Network diagram - {depth} hidden layer{'s' if depth != 1 else ''}, "
            f"{config.mode.activation} output"
        )

    # ---------------------------------------------------------------- training

    def _start_training(self) -> None:
        if self._bundle is None:
            QMessageBox.information(
                self, "No data yet", "Prepare a dataset in the Data tab first."
            )
            self.tabs.setCurrentIndex(DATA_TAB)
            return
        if self._worker is not None:
            return

        config = self.architecture_panel.config()
        blocking = [
            w
            for w in validate(config)
            if "needs at least" in w or "should have exactly" in w
        ]
        if blocking:
            QMessageBox.warning(self, "Invalid architecture", "\n".join(blocking))
            return

        request = TrainingRequest(
            config=config,
            data=self._bundle,
            epochs=self.training_panel.epochs_spin.value(),
            batch_size=self.training_panel.batch_spin.value(),
            shuffle=self.training_panel.shuffle_check.isChecked(),
            early_stopping=self.training_panel.early_check.isChecked(),
            patience=self.training_panel.patience_spin.value(),
        )
        self._total_epochs = request.epochs
        self._history = {}

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_running(True)
        self.training_panel.set_export_enabled(False)
        self.predict_panel.set_model_ready(False)
        self.curves.clear()
        self.canvas.set_training(True)
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit("Training...")

        self._worker = TrainingWorker(request, parent=self)
        self._worker.message.connect(self.training_panel.append_log)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.succeeded.connect(self._on_training_done)
        self._worker.failed.connect(self._on_training_failed)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    def _stop_training(self) -> None:
        for worker in (self._worker, self._cv_worker):
            if worker:
                worker.stop()
                self.status.emit("Stopping after the current batch...")

    def _on_epoch(self, epoch: int, logs: dict) -> None:
        for key, value in logs.items():
            self._history.setdefault(key, []).append(value)
        self.training_panel.show_epoch(epoch, self._total_epochs, logs)
        if epoch == 1 or epoch % 2 == 0:
            self.curves.update_history(self._history)
        parts = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.training_panel.append_log(f"epoch {epoch:>4} | {parts}")

    def _on_training_done(self, result: dict) -> None:
        self._model = result["model"]
        self._history = result.get("history", self._history)
        self.curves.update_history(self._history)
        self.canvas.set_training(False)
        self.training_panel.set_running(False)
        self.training_panel.set_export_enabled(True)
        self.predict_panel.set_model_ready(True)

        scores = result.get("final_scores", {})
        if scores:
            summary = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            self.training_panel.append_log(f"\nFinal validation: {summary}")
            self.status.emit(f"Done - {summary}")
        else:
            self.status.emit("Training finished")
        if result.get("stopped"):
            self.training_panel.append_log("Run interrupted by the user.")

    def _on_training_failed(self, message: str) -> None:
        self.canvas.set_training(False)
        self.training_panel.set_running(False)
        self.training_panel.append_log(f"\nTRAINING FAILED\n{message}")
        self.status.emit("Training failed")
        QMessageBox.critical(self, "Training failed", message.split("\n\n")[0])

    def _cleanup_worker(self) -> None:
        if self._worker:
            self._worker.deleteLater()
        self._worker = None

    # -------------------------------------------------------- cross-validation

    def _start_crossval(self) -> None:
        spec = self.data_panel.current_spec()
        if spec is None:
            QMessageBox.information(
                self, "No data yet", "Prepare a dataset in the Data tab first."
            )
            self.tabs.setCurrentIndex(DATA_TAB)
            return
        if self._worker is not None or self._cv_worker is not None:
            return

        k = self.training_panel.k_spin.value()
        epochs = self.training_panel.epochs_spin.value()
        if not self._confirm_cost(k, epochs):
            return

        request = CrossValRequest(
            spec=spec,
            template=self.architecture_panel.config(),
            k=k,
            epochs=epochs,
            batch_size=self.training_panel.batch_spin.value(),
            shuffle=self.training_panel.shuffle_check.isChecked(),
            early_stopping=self.training_panel.early_check.isChecked(),
            patience=self.training_panel.patience_spin.value(),
        )

        # A cross-validation produces an estimate, not a model. Any model left
        # over from an earlier run is unrelated to it, so drop it now rather
        # than leaving Predict and Save pointing at something stale.
        self._model = None
        self._history = {}
        self.predict_panel.set_model_ready(False)
        self.training_panel.set_export_enabled(False)

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_cross_validating(True)
        self.curves.clear()
        self.canvas.set_training(True)
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit(f"Cross-validating over {k} folds...")
        self.training_panel.append_log(
            f"Cross-validation: {k} folds x {epochs} epochs.\n"
            "The k models are discarded - only the mean and spread are kept.\n"
        )

        self._cv_worker = CrossValWorker(request, parent=self)
        self._cv_worker.message.connect(self.training_panel.append_log)
        self._cv_worker.fold_done.connect(self._on_fold)
        self._cv_worker.succeeded.connect(self._on_crossval_done)
        self._cv_worker.failed.connect(self._on_crossval_failed)
        self._cv_worker.finished.connect(self._cleanup_cv_worker)
        self._cv_worker.start()

    def _confirm_cost(self, k: int, epochs: int) -> bool:
        answer = QMessageBox.question(
            self,
            "Cross-validation cost",
            f"This trains {k} separate models of {epochs} epochs each - "
            f"roughly {k} times a normal run.\n\n"
            "None of them is kept. The result is an estimate of the "
            "architecture, not a trained model.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_fold(self, index: int, total: int, entry: dict) -> None:
        self.training_panel.show_fold(index, total, entry)
        parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(entry["scores"].items()))
        self.training_panel.append_log(
            f"fold {index}/{total} done ({entry['epochs_run']} epochs) | {parts}"
        )

    def _on_crossval_done(self, result: dict) -> None:
        self.canvas.set_training(False)
        self.training_panel.set_cross_validating(False)
        if not result.get("folds"):
            self.training_panel.show_crossval_summary("cancelled")
            self.status.emit("Cross-validation cancelled")
            return

        self.curves.show_folds(result)
        self.training_panel.append_log("\n" + format_summary(result))

        metric = result.get("metric")
        if metric:
            stats = result["summary"][metric]
            headline = (
                f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
                f"over {result['completed']} folds"
            )
            self.training_panel.show_crossval_summary(headline)
            self.status.emit(f"Cross-validation done - {headline}")
        else:
            self.training_panel.show_crossval_summary("done")

        self.training_panel.append_log(
            "\nNo model was kept. Train once on the full split to get one."
        )

    def _on_crossval_failed(self, message: str) -> None:
        self.canvas.set_training(False)
        self.training_panel.set_cross_validating(False)
        self.training_panel.show_crossval_summary("failed")
        self.training_panel.append_log(f"\nCROSS-VALIDATION FAILED\n{message}")
        self.status.emit("Cross-validation failed")
        QMessageBox.critical(
            self, "Cross-validation failed", message.split("\n\n")[0]
        )

    def _cleanup_cv_worker(self) -> None:
        if self._cv_worker:
            self._cv_worker.deleteLater()
        self._cv_worker = None

    # --------------------------------------------------------------- inference

    def _predict(self, values: dict) -> None:
        if self._model is None or self._bundle is None:
            self.predict_panel.show_result("Train the network first")
            return
        try:
            x = self._bundle.transform_row(values)
            prediction = self._model.predict(x, verbose=0)[0]
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.predict_panel.show_result(f"Could not predict: {exc}")
            return
        self.predict_panel.show_result(self._bundle.describe_prediction(prediction))

    # ------------------------------------------------------------------ export

    def _export_model(self) -> None:
        if self._model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model", "model.keras", "Keras model (*.keras)"
        )
        if not path:
            return
        if not path.endswith(".keras"):
            path += ".keras"
        try:
            self._model.save(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Could not save", str(exc))
            return
        self.status.emit(f"Model saved to {path}")

    # ------------------------------------------------------------------ window

    def shutdown(self) -> None:
        for worker in (self._worker, self._cv_worker):
            if worker and worker.isRunning():
                worker.stop()
                worker.wait(5000)
