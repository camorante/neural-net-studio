"""Application shell: two independent workspaces, one per kind of network."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .dense_workspace import DenseWorkspace
from .vision_workspace import VisionWorkspace

DENSE, VISION = 0, 1


class MainWindow(QMainWindow):
    """A thin shell. All the work lives inside the two workspaces.

    A convolutional network is not a later stage of a dense one, so the two
    get their own top-level tab and their own set of stages underneath.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Net Studio - interactive Keras playground")
        self.resize(1600, 960)

        self.dense = DenseWorkspace()
        self.vision = VisionWorkspace()

        workspaces = QTabWidget()
        workspaces.setObjectName("Workspaces")
        workspaces.addTab(self.dense, "Dense network  (NN)")
        workspaces.addTab(self.vision, "Convolutional  (CNN)")
        workspaces.currentChanged.connect(self._on_workspace_changed)
        self.workspaces = workspaces

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 8, 10, 6)
        layout.addWidget(workspaces)
        self.setCentralWidget(wrapper)

        self.dense.status.connect(self.statusBar().showMessage)
        self.vision.status.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "Dense network: prepare a dataset to begin. "
            "Switch to Convolutional for images."
        )

    def _on_workspace_changed(self, index: int) -> None:
        self.statusBar().showMessage(
            "Convolutional workspace - build an image dataset to begin"
            if index == VISION
            else "Dense workspace - prepare a tabular dataset to begin"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.dense.shutdown()
        self.vision.shutdown()
        event.accept()
