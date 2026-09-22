"""The convolutional workspace: four stages plus their own visuals.

A CNN is not a later step of a dense network, it is a different path. So this
workspace owns its own data, its own model, its own workers and its own
diagrams, and shares nothing with the dense side but the theme.
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

from ..core import vision as vz
from ..core.resnet import ResNetConfig
from ..core.vision_trainer import VisionRequest, format_comparison
from .image_grid import ImageGrid
from .plots import LearningCurves
from .resnet_canvas import ResNetCanvas
from .vision.architecture_panel import VisionArchitecturePanel
from .vision.images_panel import VisionImagesPanel
from .vision.predict_panel import VisionPredictPanel
from .vision.training_panel import VisionTrainingPanel
from .widgets import Card
from .workers import DatasetWorker, SkipComparisonWorker, VisionWorker

IMAGES_TAB, ARCHITECTURE_TAB, TRAINING_TAB, PREDICT_TAB = range(4)


class VisionWorkspace(QWidget):
    """Stage tabs on the left, image grid and diagrams on the right."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle: vz.ImageBundle | None = None
        self._model = None
        self._data_worker: DatasetWorker | None = None
        self._train_worker = None
        self._histories: dict = {}
        self._total_epochs = 0
        self._rng = np.random.default_rng()

        self.images_panel = VisionImagesPanel()
        self.architecture_panel = VisionArchitecturePanel()
        self.training_panel = VisionTrainingPanel()
        self.predict_panel = VisionPredictPanel()

        self.canvas = ResNetCanvas()
        self.grid = ImageGrid()
        self.curves = LearningCurves()

        self._build_layout()
        self._connect()
        self._sync_plan()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self.images_panel, "1. Images")
        tabs.addTab(self.architecture_panel, "2. Architecture")
        tabs.addTab(self.training_panel, "3. Training")
        tabs.addTab(self.predict_panel, "4. Predict")
        tabs.setMinimumWidth(400)
        self.tabs = tabs

        self.diagram_card = Card("Convolutional stack")
        self.diagram_card.add(self.canvas, 1)
        images_card = Card("Images")
        images_card.add(self.grid, 1)
        curves_card = Card("Learning curves")
        curves_card.add(self.curves, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.diagram_card)
        right.addWidget(images_card)
        right.addWidget(curves_card)
        for index, factor in ((0, 3), (1, 3), (2, 2)):
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
        self.images_panel.build_requested.connect(self._build_dataset)
        self.images_panel.cancel_requested.connect(self._cancel_dataset)
        self.images_panel.plan_changed.connect(self._sync_plan)

        self.architecture_panel.config_changed.connect(self._on_config_changed)
        self.architecture_panel.preset_requested.connect(self._apply_preset)

        self.training_panel.train_requested.connect(self._train)
        self.training_panel.compare_requested.connect(self._compare)
        self.training_panel.stop_requested.connect(self._stop)

        self.predict_panel.predict_requested.connect(self._show_predictions)

    # -------------------------------------------------------------- dataset

    def _sync_plan(self) -> None:
        """Keep the architecture tab honest before any data exists."""
        if self._bundle is not None:
            self.architecture_panel.set_data_shape(
                self._bundle.input_shape, self._bundle.n_classes, locked=True
            )
            return
        size = self.images_panel.planned_image_size()
        self.architecture_panel.set_data_shape(
            (size, size, 3), self.images_panel.planned_n_classes(), locked=False
        )

    def _build_dataset(self, builder, description: str) -> None:
        if self._data_worker or self._train_worker:
            return
        self.images_panel.set_busy(True)
        self.training_panel.append_log(description)
        self.status.emit(description)

        self._data_worker = DatasetWorker(builder, description, parent=self)
        self._data_worker.message.connect(self.status.emit)
        self._data_worker.progress.connect(self._on_dataset_progress)
        self._data_worker.succeeded.connect(self._on_dataset_ready)
        self._data_worker.cancelled.connect(self._on_dataset_cancelled)
        self._data_worker.failed.connect(self._on_dataset_failed)
        self._data_worker.finished.connect(self._cleanup_data_worker)
        self._data_worker.start()

    def _cancel_dataset(self) -> None:
        if self._data_worker:
            self._data_worker.stop()
            self.images_panel.show_status("Stopping...")
            self.status.emit("Cancelling the dataset build...")

    def _on_dataset_progress(self, done: int, total: int, label: str,
                             speed: float) -> None:
        self.images_panel.show_progress(done, total, label, speed)
        if total > 0 and label.startswith("Downloading"):
            self.status.emit(
                f"{label}: {done / 1e6:.0f} of {total / 1e6:.0f} MB"
            )

    def _on_dataset_cancelled(self) -> None:
        # The panel knows which source was running, and only the download
        # leaves anything behind - saying otherwise would be a lie.
        self.images_panel.show_cancelled()
        self.training_panel.append_log("Dataset build cancelled by the user.")
        self.status.emit("Dataset build cancelled")

    def _on_dataset_ready(self, bundle) -> None:
        self._bundle = bundle
        self._model = None
        self._histories = {}

        self.images_panel.show_summary(bundle.describe())
        self.training_panel.append_log(bundle.describe())
        self.training_panel.reset_metrics()
        self.predict_panel.set_ready(False)
        self.curves.clear()

        images, labels, _ = bundle.sample(12, self._rng)
        self.grid.show_samples(images, labels, bundle.class_names)
        self._sync_plan()
        self.status.emit(f"{bundle.name} ready")
        self.tabs.setCurrentIndex(ARCHITECTURE_TAB)

    def _on_dataset_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nDATASET FAILED\n{message}")
        self.status.emit("Could not build the dataset")
        QMessageBox.critical(
            self, "Could not build the dataset", message.split("\n\n")[0]
        )

    def _cleanup_data_worker(self) -> None:
        if self._data_worker:
            self._data_worker.deleteLater()
        self._data_worker = None
        self.images_panel.set_busy(False)

    # --------------------------------------------------------------- config

    def _on_config_changed(self) -> None:
        panel = self.architecture_panel
        panel.refresh_summary()
        self.training_panel.set_compare_enabled(panel.is_scratch())

        if panel.is_scratch():
            config = panel.scratch_config()
            skip = config.use_skip
            self.canvas.set_architecture(
                config.summary_columns(), skip, config.depth,
                "residual" if skip else "plain, no shortcut",
            )
            self.diagram_card.set_title(
                f"Convolutional stack - {config.depth} layers, "
                f"{'residual' if skip else 'plain'}"
            )
            return

        config = panel.transfer_config()
        height, width, _ = config.input_shape
        self.canvas.set_architecture(
            [
                {"name": "Input", "detail": f"{height}x{width}x3", "kind": "input"},
                {
                    "name": config.backbone,
                    "detail": "frozen, ImageNet" if config.freeze_backbone
                    else "fine-tuning",
                    "kind": "hidden",
                },
                {
                    "name": "Your head",
                    "detail": f"{config.head_units} units -> {config.n_classes}",
                    "kind": "output",
                },
            ],
            use_skip=True,
            depth=50,
            caption="pretrained backbone",
        )
        self.diagram_card.set_title(f"Convolutional stack - {config.backbone}")

    def _apply_preset(self, epochs: int, batch: int) -> None:
        self.training_panel.apply_preset(epochs, batch)
        self.training_panel.append_log(
            "Deep preset applied: 50 weight layers, stem 8 filters, 18 epochs.\n"
            "Press 'Compare skip on/off'. Watch the TRAINING accuracy of the "
            "plain arm - it will not reach 1.0, which is the degradation problem.\n"
            "Expect roughly 2 minutes per arm with 900 images at 32px."
        )
        self.tabs.setCurrentIndex(TRAINING_TAB)

    # -------------------------------------------------------------- training

    def _request(self) -> VisionRequest | None:
        if self._bundle is None:
            QMessageBox.information(
                self, "No images yet", "Build an image dataset in the Images tab first."
            )
            self.tabs.setCurrentIndex(IMAGES_TAB)
            return None
        return VisionRequest(
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
        self._start(VisionWorker(request, parent=self), request.epochs, single=True)

    def _compare(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        if not isinstance(request.config, ResNetConfig):
            return
        answer = QMessageBox.question(
            self,
            "Two training runs",
            f"This trains the same {request.config.depth}-layer stack twice - "
            "once with skip connections, once without - from identical initial "
            f"weights.\n\nThat is {request.epochs} epochs x 2.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start(
            SkipComparisonWorker(request, parent=self), request.epochs * 2, single=False
        )

    def _start(self, worker, total_epochs: int, single: bool) -> None:
        self._train_worker = worker
        self._total_epochs = total_epochs
        self._histories = {}

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_running(True)
        self.images_panel.set_busy(True)
        self.architecture_panel.set_running(True)
        self.predict_panel.set_ready(False)
        self.curves.clear()
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit("Training the convolutional model...")

        worker.message.connect(self.training_panel.append_log)
        worker.epoch_done.connect(self._on_epoch)
        worker.failed.connect(self._on_train_failed)
        worker.finished.connect(self._cleanup_train_worker)
        worker.succeeded.connect(
            self._on_train_done if single else self._on_compare_done
        )
        worker.start()

    def _stop(self) -> None:
        if self._train_worker:
            self._train_worker.stop()
            self.status.emit("Stopping after the current batch...")

    def _on_epoch(self, epoch: int, logs: dict, label: str) -> None:
        history = self._histories.setdefault(label or "run", {})
        for name, value in logs.items():
            history.setdefault(name, []).append(value)

        done = sum(len(h.get("loss", [])) for h in self._histories.values())
        self.training_panel.show_progress(done, self._total_epochs, epoch, logs, label)
        self.curves.show_arms(self._histories)
        parts = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
        prefix = f"{label} " if label else ""
        self.training_panel.append_log(f"{prefix}epoch {epoch:>3} | {parts}")

    def _on_train_done(self, result: dict) -> None:
        self._model = result["model"]
        self._histories = {result["label"]: result["history"]}
        self.curves.show_arms(self._histories)
        scores = result.get("final_scores", {})
        if scores:
            summary = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            self.training_panel.append_log(f"\nFinal validation: {summary}")
            self.status.emit(f"Done - {summary}")
        self.predict_panel.set_ready(True)
        self._show_predictions(12)

    def _on_compare_done(self, result: dict) -> None:
        arms = result.get("arms") or []
        self._histories = {arm["label"]: arm["history"] for arm in arms}
        self.curves.show_arms(self._histories)
        self.training_panel.append_log("\n" + format_comparison(result))

        residual = next((a for a in arms if a["label"] == "residual"), None)
        if residual:
            # The residual arm is the one worth keeping around to predict with.
            self._model = residual["model"]
            self.predict_panel.set_ready(True)
            self._show_predictions(12)
        self.status.emit("Skip-connection comparison finished")

    def _on_train_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nTRAINING FAILED\n{message}")
        self.status.emit("Convolutional training failed")
        QMessageBox.critical(self, "Training failed", message.split("\n\n")[0])

    def _cleanup_train_worker(self) -> None:
        if self._train_worker:
            self._train_worker.deleteLater()
        self._train_worker = None
        self.training_panel.set_running(False)
        self.images_panel.set_busy(False)
        self.architecture_panel.set_running(False)
        self.training_panel.set_compare_enabled(self.architecture_panel.is_scratch())

    # -------------------------------------------------------------- inference

    def _show_predictions(self, count: int = 12) -> None:
        if self._model is None or self._bundle is None:
            self.predict_panel.show_result("Train the network first.")
            return
        images, truths, _ = self._bundle.sample(int(count), self._rng)
        try:
            probabilities = self._model.predict(images, verbose=0)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.predict_panel.show_result("Could not predict", str(exc))
            return

        self.grid.show_predictions(
            images, truths, probabilities, self._bundle.class_names
        )
        predicted = np.argmax(probabilities, axis=1)
        hits = int((predicted == np.asarray(truths)).sum())
        total = len(truths)
        self.predict_panel.show_result(
            f"{hits} of {total} correct in this batch",
            "A batch this small says little on its own - the validation accuracy "
            "in the Training tab is the number to trust.",
        )

    # ----------------------------------------------------------------- window

    def shutdown(self) -> None:
        for worker in (self._train_worker, self._data_worker):
            if worker and worker.isRunning():
                if hasattr(worker, "stop"):
                    worker.stop()
                worker.wait(5000)
