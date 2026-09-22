"""CNN stage 1: choose an image source and build the dataset."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core import vision as vz
from ..widgets import compact_combo, hint, scrollable

SOURCE_LABELS = (
    "Synthetic shapes - generated offline",
    "CIFAR-10 - downloads ~170 MB once",
    "Image folder - one subfolder per class",
)

SYNTHETIC, CIFAR, FOLDER = 0, 1, 2


def _eta(seconds: float) -> str:
    """A rough remaining time a person can act on, not a precise one."""
    if seconds < 90:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(round(minutes))} min"
    return f"{minutes / 60:.1f} h"


class VisionImagesPanel(QWidget):
    """Source selection and dataset construction."""

    build_requested = pyqtSignal(object, str)
    cancel_requested = pyqtSignal()
    plan_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder_path = ""

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_source_group())
        inner.addWidget(self._build_summary_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

    # ------------------------------------------------------------------- build

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_LABELS))
        compact_combo(self.source_combo)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        row.addWidget(self.source_combo, 1)
        layout.addLayout(row)

        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self._build_synthetic_page())
        self.source_stack.addWidget(self._build_cifar_page())
        self.source_stack.addWidget(self._build_folder_page())
        layout.addWidget(self.source_stack)

        row = QHBoxLayout()
        self.build_button = QPushButton("Build dataset")
        self.build_button.setObjectName("Primary")
        self.build_button.setMinimumHeight(30)
        self.build_button.clicked.connect(self._request_build)
        row.addWidget(self.build_button, 2)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setMinimumHeight(30)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        row.addWidget(self.cancel_button, 1)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("")
        layout.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Hint")
        self.progress_label.setWordWrap(True)
        self.progress_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.progress_label)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        layout = QVBoxLayout(group)
        self.summary_label = QLabel("No images yet. Press Build dataset.")
        self.summary_label.setObjectName("Hint")
        self.summary_label.setWordWrap(True)
        self.summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.summary_label)
        layout.addWidget(
            hint(
                "Sample images appear on the right as soon as the dataset is "
                "built, so you can see what the network will actually be given."
            )
        )
        return group

    def _build_synthetic_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Images:"), 0, 0)
        self.n_images_spin = QSpinBox()
        self.n_images_spin.setRange(40, 20000)
        self.n_images_spin.setSingleStep(100)
        self.n_images_spin.setValue(900)
        grid.addWidget(self.n_images_spin, 0, 1)

        grid.addWidget(QLabel("Shape classes:"), 1, 0)
        self.n_shapes_spin = QSpinBox()
        self.n_shapes_spin.setRange(2, len(vz.SHAPE_NAMES))
        self.n_shapes_spin.setValue(3)
        self.n_shapes_spin.valueChanged.connect(self.plan_changed.emit)
        grid.addWidget(self.n_shapes_spin, 1, 1)

        grid.addWidget(QLabel("Image size:"), 2, 0)
        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems(["32", "48", "64"])
        compact_combo(self.image_size_combo, 5)
        self.image_size_combo.currentTextChanged.connect(self.plan_changed.emit)
        grid.addWidget(self.image_size_combo, 2, 1)

        grid.addWidget(QLabel("Noise:"), 3, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setValue(0.10)
        grid.addWidget(self.noise_spin, 3, 1)

        self.rotate_check = QCheckBox("Rotate")
        self.rotate_check.setChecked(True)
        self.rotate_check.setToolTip("Draw each shape at a random angle.")
        grid.addWidget(self.rotate_check, 4, 0)

        self.position_check = QCheckBox("Move")
        self.position_check.setChecked(True)
        self.position_check.setToolTip(
            "Random placement is what makes translation invariance valuable, "
            "and therefore what a convolution buys you over a dense layer."
        )
        grid.addWidget(self.position_check, 4, 1)

        grid.addWidget(
            hint(
                "Shapes are drawn with Pillow and generated in memory. "
                "Nothing is downloaded."
            ),
            5, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return page

    def _build_cifar_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Classes:"))
        self.cifar_list = QListWidget()
        self.cifar_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.cifar_list.setMaximumHeight(120)
        for name in vz.CIFAR10_CLASSES:
            QListWidgetItem(name, self.cifar_list)
        for index in (0, 3, 8):  # airplane, cat, ship
            self.cifar_list.item(index).setSelected(True)
        self.cifar_list.itemSelectionChanged.connect(self.plan_changed.emit)
        layout.addWidget(self.cifar_list)

        row = QHBoxLayout()
        row.addWidget(QLabel("Per class:"))
        self.cifar_per_class_spin = QSpinBox()
        self.cifar_per_class_spin.setRange(50, 6000)
        self.cifar_per_class_spin.setSingleStep(100)
        self.cifar_per_class_spin.setValue(400)
        row.addWidget(self.cifar_per_class_spin, 1)
        layout.addLayout(row)

        layout.addWidget(
            hint(
                "Real photographs, far harder than the shapes. The first build "
                "downloads about 170 MB from the Keras dataset mirror and caches "
                "it in ~/.keras/datasets."
            )
        )
        return page

    def _build_folder_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setObjectName("Subtle")
        self.folder_label.setWordWrap(True)
        self.folder_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        row.addWidget(self.folder_label, 1)
        pick = QPushButton("Choose...")
        pick.clicked.connect(self._choose_folder)
        row.addWidget(pick)
        layout.addLayout(row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Resize to:"))
        self.folder_size_combo = QComboBox()
        self.folder_size_combo.addItems(["32", "48", "64", "96"])
        compact_combo(self.folder_size_combo, 5)
        self.folder_size_combo.setCurrentText("64")
        self.folder_size_combo.currentTextChanged.connect(self.plan_changed.emit)
        size_row.addWidget(self.folder_size_combo, 1)
        layout.addLayout(size_row)

        layout.addWidget(
            hint(
                "Point at a folder holding one subfolder per class, for example "
                "photos/cats/ and photos/dogs/. Images are read locally."
            )
        )
        return page

    # ----------------------------------------------------------------- queries

    def planned_image_size(self) -> int:
        """Image side the next build will produce, before it exists."""
        source = self.source_combo.currentIndex()
        if source == SYNTHETIC:
            return int(self.image_size_combo.currentText())
        if source == FOLDER:
            return int(self.folder_size_combo.currentText())
        return 32

    def planned_n_classes(self) -> int:
        source = self.source_combo.currentIndex()
        if source == SYNTHETIC:
            return self.n_shapes_spin.value()
        if source == CIFAR:
            return max(2, len(self.cifar_list.selectedItems()))
        return 2

    # ----------------------------------------------------------------- actions

    def _on_source_changed(self, index: int) -> None:
        self.source_stack.setCurrentIndex(index)
        self.plan_changed.emit()

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose an image folder")
        if path:
            self._folder_path = path
            self.folder_label.setText(path)
            self.plan_changed.emit()

    def _request_build(self) -> None:
        source = self.source_combo.currentIndex()

        if source == SYNTHETIC:
            spec = vz.SyntheticSpec(
                n_images=self.n_images_spin.value(),
                image_size=int(self.image_size_combo.currentText()),
                n_classes=self.n_shapes_spin.value(),
                noise=float(self.noise_spin.value()),
                rotate=self.rotate_check.isChecked(),
                vary_position=self.position_check.isChecked(),
            )
            self.build_requested.emit(
                lambda on_progress, should_stop: vz.make_synthetic(
                    spec, on_progress, should_stop
                ),
                f"Drawing {spec.n_images} shapes at {spec.image_size}px...",
            )
            return

        if source == CIFAR:
            names = [item.text() for item in self.cifar_list.selectedItems()]
            if len(names) < 2:
                QMessageBox.information(
                    self, "Pick classes", "Select at least 2 CIFAR-10 classes."
                )
                return
            if not self._confirm_download():
                return
            per_class = self.cifar_per_class_spin.value()
            self.build_requested.emit(
                lambda on_progress, should_stop: vz.load_cifar10(
                    names,
                    per_class=per_class,
                    on_progress=on_progress,
                    should_stop=should_stop,
                ),
                f"Loading CIFAR-10 ({', '.join(names)})...",
            )
            return

        if not self._folder_path:
            QMessageBox.information(
                self, "No folder", "Choose a folder with one subfolder per class."
            )
            return
        path, size = self._folder_path, int(self.folder_size_combo.currentText())
        self.build_requested.emit(
            lambda on_progress, should_stop: vz.load_folder(
                path,
                image_size=size,
                on_progress=on_progress,
                should_stop=should_stop,
            ),
            f"Reading images from {path}...",
        )

    def _confirm_download(self) -> bool:
        """Be honest about the wait. This server really is that slow."""
        if vz.cifar10_archive_path().exists():
            return True
        answer = QMessageBox.question(
            self,
            "Download required",
            "CIFAR-10 is not on this machine yet.\n\n"
            "It is 170 MB, and the university server that hosts it serves at "
            "roughly 0.1 MB/s - so expect around 30 minutes.\n\n"
            "You will see live progress, and Cancel stops it at any point. "
            "The download resumes from where it stopped, so nothing is wasted.\n\n"
            "It is cached in ~/.keras/datasets and only downloads once.\n\n"
            "Start the download?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ public

    def set_busy(self, busy: bool) -> None:
        self.build_button.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if not busy:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("")

    def show_progress(self, done: int, total: int, label: str, speed: float) -> None:
        """Bytes for a download, items for a build - both read the same way."""
        if total <= 0:
            self.progress.setRange(0, 0)  # indeterminate: something is happening
            self.progress.setFormat("")
            self.progress_label.setText(label)
            return

        self.progress.setRange(0, 100)
        percent = min(100, int(done / total * 100))
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}%")

        if label.startswith("Downloading"):
            line = f"{label} - {done / 1e6:.1f} of {total / 1e6:.0f} MB"
            if speed > 0 and done < total:
                line += (
                    f" at {speed / 1e6:.2f} MB/s"
                    f" - about {_eta((total - done) / speed)} left"
                )
            self.progress_label.setText(line)
        else:
            self.progress_label.setText(f"{label} - {done} of {total}")

    def show_status(self, text: str) -> None:
        self.progress_label.setText(text)

    def show_cancelled(self) -> None:
        """Only the download leaves something behind to resume from."""
        if self.source_combo.currentIndex() == CIFAR:
            self.progress_label.setText(
                "Cancelled. The part already downloaded is kept - pressing "
                "Build dataset again resumes from there."
            )
        else:
            self.progress_label.setText("Cancelled. Nothing was kept.")

    def show_summary(self, text: str) -> None:
        self.summary_label.setText(text)
