"""CNN stage 2: design the stack, from scratch or on a pretrained backbone."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.resnet import (
    BACKBONES,
    ResNetConfig,
    TransferConfig,
    weights_are_cached,
)
from ..widgets import compact_combo, hint, scrollable

MODE_LABELS = (
    "From scratch - build the residual stack yourself",
    "Transfer learning - pretrained ResNet50",
)

SCRATCH, TRANSFER = 0, 1

DEEP_PRESET_EPOCHS = 18
DEEP_PRESET_BATCH = 32


class VisionArchitecturePanel(QWidget):
    """Emits a config whenever anything changes."""

    config_changed = pyqtSignal()
    preset_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_shape = (32, 32, 3)
        self._n_classes = 3

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_mode_group())
        inner.addWidget(self._build_summary_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))

    # ------------------------------------------------------------------- build

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODE_LABELS))
        compact_combo(self.mode_combo)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_scratch_page())
        self.mode_stack.addWidget(self._build_transfer_page())
        layout.addWidget(self.mode_stack)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Summary")
        layout = QVBoxLayout(group)
        self.params_label = QLabel("-")
        self.params_label.setObjectName("Metric")
        self.params_label.setWordWrap(True)
        self.params_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.params_label)

        self.shape_label = QLabel("-")
        self.shape_label.setObjectName("Hint")
        self.shape_label.setWordWrap(True)
        self.shape_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.shape_label)
        return group

    def _build_scratch_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Stages:"), 0, 0)
        self.stages_spin = QSpinBox()
        self.stages_spin.setRange(1, 5)
        self.stages_spin.setValue(3)
        self.stages_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.stages_spin, 0, 1)

        grid.addWidget(QLabel("Blocks/stage:"), 1, 0)
        self.blocks_spin = QSpinBox()
        self.blocks_spin.setRange(1, 8)
        self.blocks_spin.setValue(1)
        self.blocks_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.blocks_spin, 1, 1)

        grid.addWidget(QLabel("Stem filters:"), 2, 0)
        self.filters_spin = QSpinBox()
        self.filters_spin.setRange(4, 128)
        self.filters_spin.setSingleStep(4)
        self.filters_spin.setValue(16)
        self.filters_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.filters_spin, 2, 1)

        grid.addWidget(QLabel("Dropout:"), 3, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.0)
        grid.addWidget(self.dropout_spin, 3, 1)

        grid.addWidget(QLabel("Learning rate:"), 4, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.5)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        grid.addWidget(self.lr_spin, 4, 1)

        self.skip_check = QCheckBox("Use skip connections")
        self.skip_check.setChecked(True)
        self.skip_check.toggled.connect(self.config_changed.emit)
        self.skip_check.setToolTip(
            "Turn this off and the same convolutions stay in place - only the "
            "addition disappears. That is the ResNet ablation."
        )
        grid.addWidget(self.skip_check, 5, 0, 1, 2)

        self.augment_check = QCheckBox("Augment")
        self.augment_check.setChecked(True)
        self.augment_check.setToolTip(
            "Random flips, shifts and rotations applied during training only."
        )
        grid.addWidget(self.augment_check, 6, 0, 1, 2)

        self.deep_preset_button = QPushButton("Preset: 50 layers")
        self.deep_preset_button.setObjectName("Ghost")
        self.deep_preset_button.setMinimumHeight(30)
        self.deep_preset_button.clicked.connect(self._apply_deep_preset)
        self.deep_preset_button.setToolTip(
            "Sets stages 3, blocks per stage 8, stem filters 8. At this depth a "
            "plain stack stops being able to fit even its training data, which is "
            "the failure ResNet was built to fix. Roughly 2 minutes per arm."
        )
        grid.addWidget(self.deep_preset_button, 7, 0, 1, 2)

        grid.addWidget(
            hint(
                "Every stage after the first halves the spatial size and doubles "
                "the filters - the classic pyramid. Around 8 to 20 layers the "
                "shortcut mostly buys speed; the real degradation appears near 50."
            ),
            8, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return page

    def _build_transfer_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Backbone:"), 0, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(list(BACKBONES))
        compact_combo(self.backbone_combo, 12)
        self.backbone_combo.currentTextChanged.connect(self.config_changed.emit)
        grid.addWidget(self.backbone_combo, 0, 1)

        grid.addWidget(QLabel("Head units:"), 1, 0)
        self.head_units_spin = QSpinBox()
        self.head_units_spin.setRange(0, 512)
        self.head_units_spin.setSingleStep(16)
        self.head_units_spin.setValue(64)
        self.head_units_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.head_units_spin, 1, 1)

        grid.addWidget(QLabel("Dropout:"), 2, 0)
        self.transfer_dropout_spin = QDoubleSpinBox()
        self.transfer_dropout_spin.setRange(0.0, 0.8)
        self.transfer_dropout_spin.setSingleStep(0.05)
        self.transfer_dropout_spin.setDecimals(2)
        self.transfer_dropout_spin.setValue(0.2)
        grid.addWidget(self.transfer_dropout_spin, 2, 1)

        grid.addWidget(QLabel("Learning rate:"), 3, 0)
        self.transfer_lr_spin = QDoubleSpinBox()
        self.transfer_lr_spin.setRange(0.00001, 0.1)
        self.transfer_lr_spin.setDecimals(5)
        self.transfer_lr_spin.setSingleStep(0.0005)
        self.transfer_lr_spin.setValue(0.001)
        grid.addWidget(self.transfer_lr_spin, 3, 1)

        self.freeze_check = QCheckBox("Freeze backbone")
        self.freeze_check.setChecked(True)
        self.freeze_check.toggled.connect(self.config_changed.emit)
        self.freeze_check.setToolTip(
            "Frozen means 23 million pretrained parameters stay fixed and only "
            "your small head learns. Unfreezing on CPU is very slow."
        )
        grid.addWidget(self.freeze_check, 4, 0, 1, 2)

        self.weights_label = QLabel("")
        self.weights_label.setObjectName("Warning")
        self.weights_label.setWordWrap(True)
        self.weights_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        grid.addWidget(self.weights_label, 5, 0, 1, 2)

        grid.addWidget(
            hint(
                "Images must be at least 32x32 RGB. The backbone was trained on "
                "ImageNet, so it already knows edges, textures and shapes - you "
                "are only teaching it your classes."
            ),
            6, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return page

    # ----------------------------------------------------------------- queries

    @property
    def mode(self) -> int:
        return self.mode_combo.currentIndex()

    def is_scratch(self) -> bool:
        return self.mode == SCRATCH

    def scratch_config(self) -> ResNetConfig:
        return ResNetConfig(
            input_shape=self._input_shape,
            n_classes=self._n_classes,
            stem_filters=self.filters_spin.value(),
            stages=self.stages_spin.value(),
            blocks_per_stage=self.blocks_spin.value(),
            use_skip=self.skip_check.isChecked(),
            dropout=float(self.dropout_spin.value()),
            augment=self.augment_check.isChecked(),
            learning_rate=float(self.lr_spin.value()),
        )

    def transfer_config(self) -> TransferConfig:
        return TransferConfig(
            input_shape=self._input_shape,
            n_classes=self._n_classes,
            backbone=self.backbone_combo.currentText(),
            freeze_backbone=self.freeze_check.isChecked(),
            head_units=self.head_units_spin.value(),
            dropout=float(self.transfer_dropout_spin.value()),
            learning_rate=float(self.transfer_lr_spin.value()),
        )

    def current_config(self):
        return self.scratch_config() if self.is_scratch() else self.transfer_config()

    # ------------------------------------------------------------------ public

    def set_data_shape(self, input_shape: tuple, n_classes: int, locked: bool) -> None:
        """Tell the panel what the images look like, planned or actual."""
        self._input_shape = tuple(int(v) for v in input_shape)
        self._n_classes = int(n_classes)
        h, w, c = self._input_shape
        origin = "from the built dataset" if locked else "planned from the Images tab"
        self.shape_label.setText(
            f"Input {h}x{w}x{c}, {self._n_classes} classes ({origin})."
        )
        self.config_changed.emit()

    def set_running(self, running: bool) -> None:
        self.mode_combo.setEnabled(not running)
        self.deep_preset_button.setEnabled(not running)

    def describe(self) -> str:
        """One line about the current configuration, for the summary label."""
        if self.is_scratch():
            config = self.scratch_config()
            return (
                f"{config.depth} weight layers deep, filters "
                f"{' -> '.join(str(f) for f in config.stage_filters())}"
            )
        config = self.transfer_config()
        self.weights_label.setText(
            ""
            if weights_are_cached(config.backbone)
            else f"{config.backbone} ImageNet weights are not cached yet - "
                 "the first run downloads about 98 MB."
        )
        return (
            "Frozen backbone: ~23.5M pretrained parameters stay fixed, "
            "only your head learns."
            if config.freeze_backbone
            else "Fine-tuning the whole backbone - very slow on CPU."
        )

    def refresh_summary(self) -> None:
        self.params_label.setText(self.describe())

    # ----------------------------------------------------------------- actions

    def _on_mode_changed(self, index: int) -> None:
        self.mode_stack.setCurrentIndex(index)
        self.config_changed.emit()

    def _apply_deep_preset(self) -> None:
        """The configuration measured to reproduce the degradation problem."""
        for widget in (self.stages_spin, self.blocks_spin, self.filters_spin):
            widget.blockSignals(True)
        self.stages_spin.setValue(3)
        self.blocks_spin.setValue(8)
        self.filters_spin.setValue(8)
        for widget in (self.stages_spin, self.blocks_spin, self.filters_spin):
            widget.blockSignals(False)
        self.skip_check.setChecked(True)
        self.augment_check.setChecked(False)
        self.config_changed.emit()
        self.preset_requested.emit(DEEP_PRESET_EPOCHS, DEEP_PRESET_BATCH)
