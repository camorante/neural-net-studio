"""Application shell: five independent workspaces, one per kind of network."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .autoencoder_workspace import AutoencoderWorkspace
from .dense_workspace import DenseWorkspace
from .sequence_workspace import SequenceWorkspace
from .transformer_workspace import TransformerWorkspace
from .vision_workspace import VisionWorkspace

DENSE, VISION, AUTOENCODER, SEQUENCE, TRANSFORMER = 0, 1, 2, 3, 4

OPENING_MESSAGE = {
    DENSE: "Dense workspace - prepare a tabular dataset to begin",
    VISION: "Convolutional workspace - build an image dataset to begin",
    AUTOENCODER: "Autoencoder workspace - build an image dataset to begin. No labels are used here.",
    SEQUENCE: "Sequence workspace - generate a task to begin. Ask whether order matters before anything else.",
    TRANSFORMER: "Transformer workspace - generate a token task. The attention map is tested, not just drawn.",
}


class MainWindow(QMainWindow):
    """A thin shell. All the work lives inside the five workspaces.

    A convolutional network is not a later stage of a dense one, an
    autoencoder is not a later stage of either - it learns without labels at
    all - a recurrent network answers a question the others never ask, whether
    the order of the data carries anything, and a transformer replaces
    recurrence with attention and has to be given position explicitly. Each
    gets its own top-level tab and its own set of stages underneath.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Net Studio - interactive Keras playground")
        self.resize(1600, 960)

        self.dense = DenseWorkspace()
        self.vision = VisionWorkspace()
        self.autoencoder = AutoencoderWorkspace()
        self.sequence = SequenceWorkspace()
        self.transformer = TransformerWorkspace()
        self._all = (self.dense, self.vision, self.autoencoder, self.sequence,
                     self.transformer)

        workspaces = QTabWidget()
        workspaces.setObjectName("Workspaces")
        workspaces.addTab(self.dense, "Dense network  (NN)")
        workspaces.addTab(self.vision, "Convolutional  (CNN)")
        workspaces.addTab(self.autoencoder, "Autoencoder  (no labels)")
        workspaces.addTab(self.sequence, "Sequences  (RNN)")
        workspaces.addTab(self.transformer, "Transformer  (attention)")
        workspaces.currentChanged.connect(self._on_workspace_changed)
        self.workspaces = workspaces

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 8, 10, 6)
        layout.addWidget(workspaces)
        self.setCentralWidget(wrapper)

        for workspace in self._all:
            workspace.status.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "Dense network: prepare a dataset to begin. "
            "Switch to Convolutional for images."
        )

    def _on_workspace_changed(self, index: int) -> None:
        self.statusBar().showMessage(OPENING_MESSAGE.get(index, ""))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for workspace in self._all:
            workspace.shutdown()
        event.accept()
