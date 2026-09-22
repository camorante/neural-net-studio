"""The autoencoder workspace: four stages, its own model, its own visuals.

Learning without labels is not a later step of learning with them, so this
workspace owns everything it needs and shares only the theme, the plotting
widgets and the image-source panel - which knows about `core.vision` and
nothing about any particular kind of network.
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

from ..core import autoencoder as ae
from ..core import vision as vz
from ..core.autoencoder_trainer import (
    SWEEP_SIZES,
    AutoencoderRequest,
    AutoencoderRun,
    format_anomalies,
    format_run,
    format_sweep,
)
from . import theme
from .autoencoder.architecture_panel import AutoencoderArchitecturePanel
from .autoencoder.reconstruct_panel import AutoencoderReconstructPanel
from .autoencoder.training_panel import AutoencoderTrainingPanel
from .autoencoder_canvas import AutoencoderCanvas
from .image_source import ImageSourcePanel
from .pair_grid import ReconstructionGrid
from .plots import LearningCurves
from .widgets import Card
from .workers import AutoencoderWorker, DatasetWorker, LatentSweepWorker

IMAGES_TAB, ARCHITECTURE_TAB, TRAINING_TAB, RECONSTRUCT_TAB = range(4)

# Ranking every validation image by error means one forward pass per image.
# On a large CIFAR slice that is slow enough to feel like a freeze, so the
# ranking runs on a sample and the readout says so rather than pretending.
MAX_RANKED = 600


class AutoencoderWorkspace(QWidget):
    """Stage tabs on the left, the shape and the reconstructions on the right."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle: vz.ImageBundle | None = None
        self._model = None
        self._encoder = None
        self._split: ae.AnomalySplit | None = None
        self._x_val_used: np.ndarray | None = None
        self._config: ae.AutoencoderConfig | None = None
        self._data_worker: DatasetWorker | None = None
        self._train_worker = None
        self._histories: dict = {}
        self._total_epochs = 0
        self._baseline = float("nan")
        self._sweep_arms: list = []
        self._rng = np.random.default_rng()

        self.images_panel = ImageSourcePanel()
        self.architecture_panel = AutoencoderArchitecturePanel()
        self.training_panel = AutoencoderTrainingPanel()
        self.reconstruct_panel = AutoencoderReconstructPanel()

        self.canvas = AutoencoderCanvas()
        self.grid = ReconstructionGrid()
        self.curves = LearningCurves()

        self._build_layout()
        self._connect()
        self._sync_plan()
        self._on_config_changed()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self.images_panel, "1. Images")
        tabs.addTab(self.architecture_panel, "2. Architecture")
        tabs.addTab(self.training_panel, "3. Training")
        tabs.addTab(self.reconstruct_panel, "4. Reconstruct")
        tabs.setMinimumWidth(400)
        self.tabs = tabs

        self.diagram_card = Card("Autoencoder shape")
        self.diagram_card.add(self.canvas, 1)
        self.grid_card = Card("Originals and reconstructions")
        self.grid_card.add(self.grid, 1)
        curves_card = Card("Learning curves")
        curves_card.add(self.curves, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.diagram_card)
        right.addWidget(self.grid_card)
        right.addWidget(curves_card)
        for index, factor in ((0, 3), (1, 4), (2, 2)):
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
        self.training_panel.sweep_requested.connect(self._sweep)
        self.training_panel.stop_requested.connect(self._stop)

        self.reconstruct_panel.reconstruct_requested.connect(self._show_reconstructions)
        self.reconstruct_panel.hardest_requested.connect(self._show_hardest)
        self.reconstruct_panel.anomaly_requested.connect(self._score_anomalies)

    # ----------------------------------------------------------------- dataset

    def _sync_plan(self) -> None:
        if self._bundle is not None:
            self.architecture_panel.set_data_shape(
                self._bundle.input_shape, locked=True
            )
            return
        size = self.images_panel.planned_image_size()
        self.architecture_panel.set_data_shape((size, size, 3), locked=False)

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
        self.images_panel.show_cancelled()
        self.training_panel.append_log("Dataset build cancelled by the user.")
        self.status.emit("Dataset build cancelled")

    def _on_dataset_ready(self, bundle) -> None:
        self._bundle = bundle
        self._forget_model()
        self._x_val_used = bundle.x_val

        self.images_panel.show_summary(bundle.describe())
        self.training_panel.append_log(bundle.describe())
        self.training_panel.append_log(
            "The class names above are only used to decide which images to "
            "withhold. Nothing in the training loop ever sees a label."
        )
        self.training_panel.reset_metrics()
        self.training_panel.set_baseline(
            ae.mean_image_baseline(bundle.x_train, bundle.x_val)
        )
        self.curves.clear()

        self.architecture_panel.set_classes(bundle.class_names)
        self._sync_plan()
        self._show_originals()
        self.status.emit(f"{bundle.name} ready - no labels needed from here on")
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

    # ------------------------------------------------------------------ config

    def _on_config_changed(self) -> None:
        panel = self.architecture_panel
        panel.refresh_summary()
        config = panel.current_config()

        caption = (
            f"{config.n_inputs:,} numbers in, {config.latent_dim} at the waist "
            f"- a {config.compression:.0f}x squeeze"
        )
        if config.denoise:
            caption += f"   |   denoising at {config.noise_std:.2f}"

        # Say it when the picture is not the tidy taper from the textbook: for
        # a strided conv encoder the first stage usually holds MORE numbers
        # than the image did, and all the squeezing happens at one layer.
        note = ""
        columns = config.stack()
        first_hidden = next((c for c in columns if c["kind"] == "encoder"), None)
        if first_hidden and first_hidden["values"] > columns[0]["values"]:
            note = (
                f"Note: {first_hidden['name']} holds "
                f"{first_hidden['values']:,} numbers - MORE than the "
                f"{columns[0]['values']:,} in the image. The squeeze is not a "
                "gradual taper; it happens in one step, at the dense layer into "
                "the latent code."
            )
        self.canvas.set_architecture(columns, caption, note)
        self.diagram_card.set_title(
            f"Autoencoder shape - {config.variant}, latent {config.latent_dim}"
        )

    def _apply_preset(self, epochs: int, batch: int) -> None:
        self.training_panel.apply_preset(epochs, batch)
        sizes = ", ".join(str(v) for v in SWEEP_SIZES)
        self.training_panel.append_log(
            f"Sweep preset applied: {epochs} epochs per arm, batch {batch}.\n"
            f"Press 'Compare latent sizes' to train latent {sizes} from the same "
            "initial weights. Then read a column of the grid top to bottom - "
            "same picture, three different waists."
        )
        self.tabs.setCurrentIndex(TRAINING_TAB)

    # ---------------------------------------------------------------- training

    def _request(self) -> AutoencoderRequest | None:
        if self._bundle is None:
            QMessageBox.information(
                self, "No images yet",
                "Build an image dataset in the Images tab first.",
            )
            self.tabs.setCurrentIndex(IMAGES_TAB)
            return None

        problems = self.architecture_panel.problems()
        if problems:
            QMessageBox.information(
                self, "The autoencoder cannot be built", "\n\n".join(problems)
            )
            self.tabs.setCurrentIndex(ARCHITECTURE_TAB)
            return None

        return AutoencoderRequest(
            data=self._bundle,
            config=self.architecture_panel.current_config(),
            epochs=self.training_panel.epochs_spin.value(),
            batch_size=self.training_panel.batch_spin.value(),
            early_stopping=self.training_panel.early_check.isChecked(),
            anomaly_class=self.architecture_panel.anomaly_class(),
        )

    def _prepare(self, request: AutoencoderRequest) -> bool:
        """Work out the arrays and the give-up floor before anything trains."""
        try:
            x_train, x_val, split = AutoencoderRun(request).arrays()
        except ae.AutoencoderError as exc:
            QMessageBox.information(self, "Cannot hold that class out", str(exc))
            self.tabs.setCurrentIndex(ARCHITECTURE_TAB)
            return False

        self._split = split
        self._x_val_used = x_val
        self._config = request.config
        self._baseline = ae.mean_image_baseline(x_train, x_val)
        self.training_panel.set_baseline(self._baseline)
        return True

    def _train(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        if not self._prepare(request):
            return
        self._start(
            AutoencoderWorker(request, parent=self), request.epochs, single=True
        )

    def _sweep(self) -> None:
        request = self._request()
        if request is None or self._train_worker:
            return
        if not self._prepare(request):
            return

        sizes = [s for s in SWEEP_SIZES if s < request.config.n_inputs]
        if len(sizes) < 2:
            QMessageBox.information(
                self, "Images too small",
                "These images are too small for the sweep's latent sizes to all "
                "be bottlenecks. Use a larger image size.",
            )
            return

        answer = QMessageBox.question(
            self,
            f"{len(sizes)} training runs",
            f"This trains the same autoencoder {len(sizes)} times - at latent "
            f"{', '.join(str(s) for s in sizes)} - from identical initial "
            f"weights.\n\nThat is {request.epochs} epochs x {len(sizes)}.\n\n"
            "Everything else stays fixed, so the width of the waist is the only "
            "variable.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._start(
            LatentSweepWorker(request, sizes, parent=self),
            request.epochs * len(sizes),
            single=False,
        )

    def _start(self, worker, total_epochs: int, single: bool) -> None:
        self._train_worker = worker
        self._total_epochs = total_epochs
        self._histories = {}
        self._sweep_arms = []
        self._model = None
        self._encoder = None

        self.training_panel.clear_log()
        self.training_panel.reset_metrics()
        self.training_panel.set_running(True)
        self.images_panel.set_busy(True)
        self.architecture_panel.set_running(True)
        self.reconstruct_panel.set_ready(False)
        self.curves.clear()
        self.tabs.setCurrentIndex(TRAINING_TAB)
        self.status.emit("Training the autoencoder - no labels in this run")

        worker.message.connect(self.training_panel.append_log)
        worker.epoch_done.connect(self._on_epoch)
        worker.failed.connect(self._on_train_failed)
        worker.finished.connect(self._cleanup_train_worker)
        worker.succeeded.connect(
            self._on_train_done if single else self._on_sweep_done
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
        self.curves.show_arms(self._histories, metric="mae", baseline=self._baseline)
        parts = " ".join(f"{k}={v:.5f}" for k, v in logs.items())
        prefix = f"{label} " if label else ""
        self.training_panel.append_log(f"{prefix}epoch {epoch:>3} | {parts}")

    def _on_train_done(self, result: dict) -> None:
        self._model = result["model"]
        self._encoder = result["encoder"]
        self._histories = {result["label"]: result["history"]}
        self._baseline = result.get("baseline", self._baseline)
        self.training_panel.set_baseline(self._baseline)
        self.curves.show_arms(self._histories, metric="mae", baseline=self._baseline)
        self.training_panel.append_log("\n" + format_run(result))

        self.reconstruct_panel.set_ready(True)
        self.reconstruct_panel.set_anomaly_available(self._split is not None)
        self.status.emit(
            f"Done - val MSE {result.get('final_scores', {}).get('loss', float('nan')):.5f}"
        )
        self._show_reconstructions(self.reconstruct_panel.count_spin.value())

    def _on_sweep_done(self, result: dict) -> None:
        arms = result.get("arms") or []
        self._sweep_arms = arms
        self._histories = {arm["label"]: arm["history"] for arm in arms}
        if arms:
            self._baseline = arms[0].get("baseline", self._baseline)
        self.training_panel.set_baseline(self._baseline)
        self.curves.show_arms(self._histories, metric="mae", baseline=self._baseline)
        self.training_panel.append_log("\n" + format_sweep(result))

        if arms:
            # The widest arm is the one worth keeping to poke at afterwards.
            widest = max(arms, key=lambda a: a["latent_dim"])
            self._model = widest["model"]
            self._encoder = widest["encoder"]
            self.reconstruct_panel.set_ready(True)
            self.reconstruct_panel.set_anomaly_available(self._split is not None)
            self._show_sweep_rows(self.reconstruct_panel.count_spin.value())
        self.status.emit("Latent sweep finished")

    def _on_train_failed(self, message: str) -> None:
        self.training_panel.append_log(f"\nTRAINING FAILED\n{message}")
        self.status.emit("Autoencoder training failed")
        QMessageBox.critical(self, "Training failed", message.split("\n\n")[0])

    def _cleanup_train_worker(self) -> None:
        if self._train_worker:
            self._train_worker.deleteLater()
        self._train_worker = None
        self.training_panel.set_running(False)
        self.images_panel.set_busy(False)
        self.architecture_panel.set_running(False)

    # --------------------------------------------------------------- inference

    def _forget_model(self) -> None:
        """A new dataset invalidates everything trained on the old one."""
        self._model = None
        self._encoder = None
        self._split = None
        self._sweep_arms = []
        self._histories = {}
        self._baseline = float("nan")
        self.reconstruct_panel.set_ready(False)
        self.grid.clear()

    def _pick(self, count: int, source: np.ndarray) -> np.ndarray:
        total = len(source)
        if total == 0:
            return np.zeros(0, dtype="int64")
        size = int(min(max(1, count), total))
        return self._rng.choice(total, size=size, replace=False)

    def _show_originals(self) -> None:
        if self._x_val_used is None or len(self._x_val_used) == 0:
            return
        picks = self._pick(8, self._x_val_used)
        self.grid.show_rows(
            [
                {
                    "label": "original",
                    "images": self._x_val_used[picks],
                    "colour": theme.INPUT_COLOR,
                }
            ],
            "Validation images. Train an autoencoder to see what comes back out.",
        )

    def _show_reconstructions(self, count: int = 8) -> None:
        if self._model is None or self._x_val_used is None:
            self.reconstruct_panel.show_result("Train an autoencoder first.")
            return
        if self._sweep_arms:
            self._show_sweep_rows(count)
            return

        config = self._config
        picks = self._pick(count, self._x_val_used)
        clean = self._x_val_used[picks]

        rows = [
            {"label": "original", "images": clean, "colour": theme.INPUT_COLOR}
        ]
        try:
            if config is not None and config.denoise:
                noisy = ae.add_noise(clean, config.noise_std, self._rng)
                rows.append(
                    {
                        "label": f"corrupted {config.noise_std:.2f}",
                        "images": noisy,
                        "colour": theme.WARNING,
                    }
                )
                rebuilt = ae.reconstruct(self._model, noisy)
                errors = ae.reconstruction_errors(self._model, noisy, against=clean)
            else:
                rebuilt = ae.reconstruct(self._model, clean)
                errors = ae.reconstruction_errors(self._model, clean)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.reconstruct_panel.show_result("Could not reconstruct", str(exc))
            return

        latent = config.latent_dim if config else 0
        rows.append(
            {
                "label": f"latent {latent}",
                "images": rebuilt,
                "errors": errors,
                "colour": theme.SUCCESS,
            }
        )
        caption = (
            f"Same {len(picks)} images through a {latent}-number waist. "
            "The figure under each one is its own reconstruction error."
        )
        if config is not None and config.denoise:
            caption += (
                " The model was fed the corrupted row and scored against the "
                "clean one."
            )
        self.grid.show_rows(rows, caption)
        self.grid_card.set_title(f"Originals and reconstructions - latent {latent}")

        mean = float(np.mean(errors)) if len(errors) else float("nan")
        detail = f"Mean error over this batch: {mean:.5f}."
        if self._baseline == self._baseline:
            detail += (
                f" Answering with the average image would score "
                f"{self._baseline:.5f}."
            )
        self.reconstruct_panel.show_result(
            f"{len(picks)} images rebuilt through {latent} numbers", detail
        )

    def _show_sweep_rows(self, count: int = 8) -> None:
        """One row per latent size, all showing the same pictures."""
        if not self._sweep_arms or self._x_val_used is None:
            return
        picks = self._pick(count, self._x_val_used)
        clean = self._x_val_used[picks]

        rows = [
            {"label": "original", "images": clean, "colour": theme.INPUT_COLOR}
        ]
        palette = (theme.DANGER, theme.WARNING, theme.SUCCESS, theme.ACCENT)
        for index, arm in enumerate(
            sorted(self._sweep_arms, key=lambda a: a["latent_dim"])
        ):
            try:
                rebuilt = ae.reconstruct(arm["model"], clean)
                errors = ae.reconstruction_errors(arm["model"], clean)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.reconstruct_panel.show_result("Could not reconstruct", str(exc))
                return
            rows.append(
                {
                    "label": f"latent {arm['latent_dim']}",
                    "images": rebuilt,
                    "errors": errors,
                    "colour": palette[index % len(palette)],
                }
            )

        self.grid.show_rows(
            rows,
            "Read a column top to bottom: one picture, one waist per row. "
            "The narrow rows keep position and colour and lose the edges.",
        )
        self.grid_card.set_title("Reconstructions by latent size")
        self.reconstruct_panel.show_result(
            f"{len(self._sweep_arms)} waists on the same {len(picks)} images",
            "The full comparison, including the score for giving up, is in the "
            "Training log.",
        )

    def _show_hardest(self, count: int = 8) -> None:
        if self._model is None or self._x_val_used is None:
            self.reconstruct_panel.show_result("Train an autoencoder first.")
            return

        pool = self._x_val_used
        sampled = len(pool) > MAX_RANKED
        if sampled:
            pool = pool[self._pick(MAX_RANKED, pool)]

        try:
            errors = ae.reconstruction_errors(self._model, pool)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.reconstruct_panel.show_result("Could not rank the images", str(exc))
            return

        size = int(min(max(1, count), len(pool)))
        worst = np.argsort(-errors)[:size]
        images = pool[worst]
        rebuilt = ae.reconstruct(self._model, images)

        self.grid.show_rows(
            [
                {"label": "original", "images": images, "colour": theme.INPUT_COLOR},
                {
                    "label": "rebuilt",
                    "images": rebuilt,
                    "errors": errors[worst],
                    "colour": theme.DANGER,
                },
            ],
            "The images this model rebuilt worst, worst on the left.",
        )
        self.grid_card.set_title("Hardest images to reconstruct")

        detail = (
            f"Ranked {len(pool)} validation images. Worst {errors[worst].max():.5f}, "
            f"median {float(np.median(errors)):.5f}, best {errors.min():.5f}."
        )
        if sampled:
            detail += (
                f" Only {MAX_RANKED} of {len(self._x_val_used)} were ranked - "
                "one forward pass each is slow enough to freeze the window."
            )
        self.reconstruct_panel.show_result(
            f"The {size} least familiar images", detail
        )

    def _score_anomalies(self) -> None:
        if self._model is None or self._split is None:
            self.reconstruct_panel.show_result(
                "No class was held out.",
                "Choose one in the Architecture tab and train again.",
            )
            return

        split = self._split
        try:
            scores = ae.score_anomalies(self._model, split)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.reconstruct_panel.show_result("Could not score", str(exc))
            return
        if not scores:
            self.reconstruct_panel.show_result("Nothing to score.")
            return

        self.training_panel.append_log("\n" + format_anomalies(scores))

        familiar_picks = self._pick(4, split.x_val)
        strange_picks = self._pick(4, split.anomalies)
        familiar = split.x_val[familiar_picks]
        strange = split.anomalies[strange_picks]

        self.grid.show_rows(
            [
                {
                    "label": "familiar",
                    "images": familiar,
                    "colour": theme.INPUT_COLOR,
                },
                {
                    "label": "rebuilt",
                    "images": ae.reconstruct(self._model, familiar),
                    "errors": ae.reconstruction_errors(self._model, familiar),
                    "colour": theme.SUCCESS,
                },
                {
                    "label": split.anomaly_name,
                    "images": strange,
                    "colour": theme.WARNING,
                },
                {
                    "label": "rebuilt",
                    "images": ae.reconstruct(self._model, strange),
                    "errors": ae.reconstruction_errors(self._model, strange),
                    "colour": theme.DANGER,
                },
            ],
            f"Top two rows: classes the model trained on. Bottom two: "
            f"'{split.anomaly_name}', which it never saw once.",
        )
        self.grid_card.set_title(f"Familiar images vs '{split.anomaly_name}'")

        self.reconstruct_panel.show_result(
            f"ROC AUC {scores['auc']:.3f} using no labels at all",
            f"'{split.anomaly_name}' rebuilds {scores['ratio']:.2f}x worse than "
            f"the classes the model trained on. The full readout is in the "
            f"Training log.",
        )
        self.status.emit(
            f"Anomaly score for '{split.anomaly_name}': AUC {scores['auc']:.3f}"
        )

    # ------------------------------------------------------------------ window

    def shutdown(self) -> None:
        for worker in (self._train_worker, self._data_worker):
            if worker and worker.isRunning():
                if hasattr(worker, "stop"):
                    worker.stop()
                worker.wait(5000)
