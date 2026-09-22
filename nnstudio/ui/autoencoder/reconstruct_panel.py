"""Autoencoder stage 4: look at what came back out.

Four questions, four buttons. What does a reconstruction look like? Which
images does the model rebuild worst? Does the class it never saw rebuild worse
than the ones it did? And - the only one that asks about something that was
never in any dataset - what does it make of a file you just picked off disk?

That last one matters more than it looks. Anomaly detection means judging
something you have no examples of, so every other button here is answering an
easier question than the one the method is for.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import hint, scrollable


class AutoencoderReconstructPanel(QWidget):
    """Sampling controls plus the readout."""

    reconstruct_requested = pyqtSignal(int)
    hardest_requested = pyqtSignal(int)
    anomaly_requested = pyqtSignal()
    judge_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = ""

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_controls_group())
        inner.addWidget(self._build_own_image_group())
        inner.addWidget(self._build_result_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

    # ------------------------------------------------------------------- build

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Look at the output")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Images to show:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(4, 16)
        self.count_spin.setValue(8)
        row.addWidget(self.count_spin, 1)
        layout.addLayout(row)

        self.reconstruct_button = QPushButton("Show reconstructions")
        self.reconstruct_button.setObjectName("Primary")
        self.reconstruct_button.setMinimumHeight(32)
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.clicked.connect(
            lambda: self.reconstruct_requested.emit(self.count_spin.value())
        )
        layout.addWidget(self.reconstruct_button)

        self.hardest_button = QPushButton("Show the hardest to rebuild")
        self.hardest_button.setMinimumHeight(30)
        self.hardest_button.setEnabled(False)
        self.hardest_button.clicked.connect(
            lambda: self.hardest_requested.emit(self.count_spin.value())
        )
        self.hardest_button.setToolTip(
            "Ranks every validation image by reconstruction error and shows the "
            "worst. These are the images the model found least familiar."
        )
        layout.addWidget(self.hardest_button)

        self.anomaly_button = QPushButton("Score the held-out class")
        self.anomaly_button.setMinimumHeight(30)
        self.anomaly_button.setEnabled(False)
        self.anomaly_button.clicked.connect(self.anomaly_requested.emit)
        self.anomaly_button.setToolTip(
            "Only available when a class was held out of training in the "
            "Architecture tab."
        )
        layout.addWidget(self.anomaly_button)

        layout.addWidget(
            hint(
                "The number under each image is its own reconstruction error. "
                "Read a column top to bottom: same picture, different waist."
            )
        )
        return group

    def _build_own_image_group(self) -> QGroupBox:
        """The only place the app is asked about something never in a dataset.

        Which is what anomaly detection actually means: you do not have
        examples of the thing you are looking for, so every earlier button -
        which samples the validation split - is answering an easier question.
        """
        group = QGroupBox("Your own image")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.file_label = QLabel("No file chosen")
        self.file_label.setObjectName("Subtle")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        row.addWidget(self.file_label, 1)
        self.choose_button = QPushButton("Choose...")
        self.choose_button.clicked.connect(self._choose_file)
        row.addWidget(self.choose_button)
        layout.addLayout(row)

        self.judge_button = QPushButton("Judge my own image")
        self.judge_button.setMinimumHeight(30)
        self.judge_button.setEnabled(False)
        self.judge_button.clicked.connect(
            lambda: self.judge_requested.emit(self._path)
        )
        self.judge_button.setToolTip(
            "Runs your file through the trained autoencoder and reports where "
            "its error falls among every image the model knows."
        )
        layout.addWidget(self.judge_button)

        layout.addWidget(
            hint(
                "Any png, jpg or bmp. It gets resized to the dataset's size - at "
                "32x32 that throws away almost everything a photo contained, so "
                "a photograph will read as unfamiliar for that reason alone. "
                "Draw a shape in any paint program for a fair test."
            )
        )
        return group

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image to judge",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
        )
        if not path:
            return
        self._path = path
        self.file_label.setText(path)
        self._refresh_judge()

    def _refresh_judge(self) -> None:
        """Needs both a trained model and a chosen file - neither alone will do."""
        self.judge_button.setEnabled(
            bool(self._path) and self.reconstruct_button.isEnabled()
        )

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Result")
        layout = QVBoxLayout(group)

        self.result_label = QLabel("Train an autoencoder first.")
        self.result_label.setObjectName("Result")
        self.result_label.setWordWrap(True)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.result_label)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Hint")
        self.detail_label.setWordWrap(True)
        self.detail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.detail_label)
        return group

    # ------------------------------------------------------------------ public

    def set_ready(self, ready: bool) -> None:
        self.reconstruct_button.setEnabled(ready)
        self.hardest_button.setEnabled(ready)
        self._refresh_judge()
        if not ready:
            self.anomaly_button.setEnabled(False)
            self.result_label.setText("Train an autoencoder first.")
            self.detail_label.setText("")

    def set_anomaly_available(self, available: bool) -> None:
        self.anomaly_button.setEnabled(available)

    def show_result(self, headline: str, detail: str = "") -> None:
        self.result_label.setText(headline)
        self.detail_label.setText(detail)
