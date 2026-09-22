"""Qt adapters around the framework-free training run."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from ..core.crossval import CrossValRequest, CrossValRun
from ..core.trainer import TrainingRequest, TrainingRun
from ..core.vision import BuildCancelled, ImageBundle
from ..core.vision_trainer import SkipComparisonRun, VisionRequest, VisionRun


class TrainingWorker(QThread):
    """Runs one training session off the GUI thread."""

    epoch_done = pyqtSignal(int, dict)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request: TrainingRequest, parent=None):
        super().__init__(parent)
        self._run = TrainingRun(
            request,
            on_epoch=lambda epoch, logs: self.epoch_done.emit(epoch, logs),
            on_message=self.message.emit,
        )

    def stop(self) -> None:
        self._run.stop()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.succeeded.emit(self._run.run())
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class CrossValWorker(QThread):
    """Runs a k-fold cross-validation off the GUI thread."""

    fold_done = pyqtSignal(int, int, dict)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request: CrossValRequest, parent=None):
        super().__init__(parent)
        self._run = CrossValRun(
            request,
            on_fold=lambda index, total, entry: self.fold_done.emit(index, total, entry),
            on_message=self.message.emit,
        )

    def stop(self) -> None:
        self._run.stop()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.succeeded.emit(self._run.run())
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class DatasetWorker(QThread):
    """Builds or loads an image dataset off the GUI thread.

    Generating thousands of shapes, decoding a folder of photos, or pulling
    CIFAR-10 all take long enough to freeze the window if done inline. The
    CIFAR-10 server in particular runs at about 0.1 MB/s, so a build can take
    half an hour - which is unusable without progress and a way out.
    """

    progress = pyqtSignal(int, int, str, float)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, builder, description: str, parent=None):
        super().__init__(parent)
        self._builder = builder
        self._description = description
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.message.emit(self._description)
            bundle = self._builder(
                lambda done, total, label, speed: self.progress.emit(
                    done, total, label, speed
                ),
                lambda: self._stop,
            )
            self.succeeded.emit(bundle)
        except BuildCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class VisionWorker(QThread):
    """Trains one convolutional model off the GUI thread."""

    epoch_done = pyqtSignal(int, dict, str)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request: VisionRequest, parent=None):
        super().__init__(parent)
        self._run = VisionRun(
            request,
            on_epoch=lambda epoch, logs, label: self.epoch_done.emit(epoch, logs, label),
            on_message=self.message.emit,
        )

    def stop(self) -> None:
        self._run.stop()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.succeeded.emit(self._run.run())
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class SkipComparisonWorker(QThread):
    """Trains the residual and plain arms back to back."""

    epoch_done = pyqtSignal(int, dict, str)
    arm_done = pyqtSignal(str, dict)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request: VisionRequest, parent=None):
        super().__init__(parent)
        self._run = SkipComparisonRun(
            request,
            on_epoch=lambda epoch, logs, label: self.epoch_done.emit(epoch, logs, label),
            on_message=self.message.emit,
            on_arm_done=lambda label, outcome: self.arm_done.emit(label, outcome),
        )

    def stop(self) -> None:
        self._run.stop()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.succeeded.emit(self._run.run())
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
