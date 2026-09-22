"""Application shell: four independent workspaces, one per kind of network."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .autoencoder_workspace import AutoencoderWorkspace
from .dense_workspace import DenseWorkspace
from .sequence_workspace import SequenceWorkspace
from .vision_workspace import VisionWorkspace

DENSE, VISION, AUTOENCODER, SEQUENCE = 0, 1, 2, 3

OPENING_MESSAGE = {
    DENSE: "Dense workspace - prepare a tabular dataset to begin",
    VISION: "Convolutional workspace - build an image dataset to begin",
    AUTOENCODER: "Autoencoder workspace - build an image dataset to begin. No labels are used here.",
    SEQUENCE: "Sequence workspace - generate a task to begin. Ask whether order matters before anything else.",
}


class MainWindow(QMainWindow):
    """A thin shell. All the work lives inside the four workspaces.

    A convolutional network is not a later stage of a dense one, an
    autoencoder is not a later stage of either - it learns without labels at
    all - and a recurrent network answers a question the others never ask:
    whether the order of the data carries anything. Each gets its own
    top-level tab and its own set of stages underneath.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Net Studio - interactive Keras playground")
        self.resize(1600, 960)

        self.dense = DenseWorkspace()
        self.vision = VisionWorkspace()
        self.autoencoder = AutoencoderWorkspace()
        self.sequence = SequenceWorkspace()

        workspaces = QTabWidget()
        workspaces.setObjectName("Workspaces")
        workspaces.addTab(self.dense, "Dense network  (NN)")
        workspaces.addTab(self.vision, "Convolutional  (CNN)")
        workspaces.addTab(self.autoencoder, "Autoencoder  (no labels)")
        workspaces.addTab(self.sequence, "Sequences  (RNN)")
        workspaces.currentChanged.connect(self._on_workspace_changed)
        self.workspaces = workspaces

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 8, 10, 6)
        layout.addWidget(workspaces)
        self.setCentralWidget(wrapper)

        self.dense.status.connect(self.statusBar().showMessage)
        self.vision.status.connect(self.statusBar().showMessage)
        self.autoencoder.status.connect(self.statusBar().showMessage)
        self.sequence.status.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "Dense network: prepare a dataset to begin. "
            "Switch to Convolutional for images."
        )

    def _on_workspace_changed(self, index: int) -> None:
        self.statusBar().showMessage(OPENING_MESSAGE.get(index, ""))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.dense.shutdown()
        self.vision.shutdown()
        self.autoencoder.shutdown()
        self.sequence.shutdown()
        event.accept()
