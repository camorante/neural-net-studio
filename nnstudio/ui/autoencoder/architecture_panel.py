"""Autoencoder stage 2: choose the waist, and what the encoder is made of.

The latent slider is the control this whole workspace exists for. Everything
else here is a supporting knob; that one is the lesson, so it gets a slider you
drag and a live readout of what you just did to the compression ratio.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.autoencoder import (
    CONV,
    DENSE,
    VARIANT_LABELS,
    AutoencoderConfig,
    estimate_params,
    validate,
)
from ...core.autoencoder_trainer import NO_ANOMALY, SWEEP_BATCH, SWEEP_EPOCHS
from ..widgets import compact_combo, hint, scrollable

VARIANTS = (CONV, DENSE)

LATENT_MIN = 1
LATENT_MAX = 256


class AutoencoderArchitecturePanel(QWidget):
    """Emits a config whenever anything changes."""

    config_changed = pyqtSignal()
    preset_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_shape = (32, 32, 3)
        self._class_names: list = []

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(12)
        inner.addWidget(self._build_latent_group())
        inner.addWidget(self._build_encoder_group())
        inner.addWidget(self._build_unsupervised_group())
        inner.addWidget(self._build_summary_group())
        inner.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(content))
        self._sync_variant(0)

    # ------------------------------------------------------------------- build

    def _build_latent_group(self) -> QGroupBox:
        group = QGroupBox("The bottleneck")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.latent_label = QLabel("-")
        self.latent_label.setObjectName("Metric")
        self.latent_label.setWordWrap(True)
        self.latent_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.latent_label)

        row = QHBoxLayout()
        self.latent_slider = QSlider(Qt.Orientation.Horizontal)
        self.latent_slider.setRange(LATENT_MIN, LATENT_MAX)
        self.latent_slider.setValue(16)
        self.latent_slider.setPageStep(8)
        self.latent_slider.setToolTip(
            "How many numbers the whole image has to fit through. This is the "
            "only control in the app whose effect you can see with your eyes."
        )
        self.latent_slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.latent_slider, 1)

        self.latent_spin = QSpinBox()
        self.latent_spin.setRange(LATENT_MIN, LATENT_MAX)
        self.latent_spin.setValue(16)
        self.latent_spin.valueChanged.connect(self._on_spin)
        row.addWidget(self.latent_spin)
        layout.addLayout(row)

        self.sweep_button = QPushButton("Preset: sweep the bottleneck")
        self.sweep_button.setObjectName("Ghost")
        self.sweep_button.setMinimumHeight(30)
        self.sweep_button.setToolTip(
            "Sets the run up to train latent 2, 8 and 64 back to back from the "
            "same initial weights. Measured at about 25 epochs per arm, which is "
            "where the arms actually pull apart - under a minute for all three "
            "on 900 shapes at 32px."
        )
        self.sweep_button.clicked.connect(
            lambda: self.preset_requested.emit(SWEEP_EPOCHS, SWEEP_BATCH)
        )
        layout.addWidget(self.sweep_button)

        layout.addWidget(
            hint(
                "Drag it low and the reconstruction blurs; drag it as wide as the "
                "image and there is no lesson left, because copying the input "
                "becomes the cheapest answer."
            )
        )
        return group

    def _build_encoder_group(self) -> QGroupBox:
        group = QGroupBox("Encoder and decoder")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.variant_combo = QComboBox()
        self.variant_combo.addItems([VARIANT_LABELS[v] for v in VARIANTS])
        compact_combo(self.variant_combo)
        self.variant_combo.currentIndexChanged.connect(self._sync_variant)
        layout.addWidget(self.variant_combo)

        grid = QGridLayout()
        grid.addWidget(QLabel("Stages:"), 0, 0)
        self.stages_spin = QSpinBox()
        self.stages_spin.setRange(1, 3)
        self.stages_spin.setValue(2)
        self.stages_spin.valueChanged.connect(self.config_changed.emit)
        self.stages_spin.setToolTip(
            "Each convolutional stage halves the picture and doubles the "
            "filters on the way in, and the decoder undoes both on the way out."
        )
        grid.addWidget(self.stages_spin, 0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.width_stack = QStackedWidget()
        self.width_stack.addWidget(self._build_conv_page())
        self.width_stack.addWidget(self._build_dense_page())
        layout.addWidget(self.width_stack)

        grid = QGridLayout()
        grid.addWidget(QLabel("Learning rate:"), 0, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.5)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        grid.addWidget(self.lr_spin, 0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        return group

    def _build_conv_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Base filters:"), 0, 0)
        self.filters_spin = QSpinBox()
        self.filters_spin.setRange(4, 64)
        self.filters_spin.setSingleStep(4)
        self.filters_spin.setValue(16)
        self.filters_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.filters_spin, 0, 1)

        grid.addWidget(
            hint(
                "Convolutions keep the picture's geometry, so a pixel's "
                "neighbours stay its neighbours all the way to the waist."
            ),
            1, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return page

    def _build_dense_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Hidden units:"), 0, 0)
        self.units_spin = QSpinBox()
        self.units_spin.setRange(32, 1024)
        self.units_spin.setSingleStep(32)
        self.units_spin.setValue(256)
        self.units_spin.valueChanged.connect(self.config_changed.emit)
        grid.addWidget(self.units_spin, 0, 1)

        grid.addWidget(
            hint(
                "A dense encoder flattens the image first, which throws the "
                "geometry away - it has to relearn that pixel 1 sits next to "
                "pixel 2. Compare the two variants at the same latent size."
            ),
            1, 0, 1, 2,
        )
        grid.setColumnStretch(1, 1)
        return page

    def _build_unsupervised_group(self) -> QGroupBox:
        group = QGroupBox("What counts as normal")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.denoise_check = QCheckBox("Denoising: corrupt the input, score the clean one")
        self.denoise_check.toggled.connect(self._on_denoise)
        self.denoise_check.setToolTip(
            "Noise is added during training only, and the target stays clean. "
            "The network is asked to repair damage nobody described to it."
        )
        layout.addWidget(self.denoise_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Noise:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.05, 1.00)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setValue(0.25)
        self.noise_spin.setEnabled(False)
        self.noise_spin.valueChanged.connect(self.config_changed.emit)
        row.addWidget(self.noise_spin, 1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Hold out:"))
        self.anomaly_combo = QComboBox()
        self.anomaly_combo.addItem("nothing - train on every class")
        compact_combo(self.anomaly_combo, 14)
        self.anomaly_combo.currentIndexChanged.connect(self.config_changed.emit)
        self.anomaly_combo.setToolTip(
            "Remove one class from training entirely. The model never sees it, "
            "so whatever it says about that class afterwards is a statement "
            "about unfamiliarity - which is anomaly detection."
        )
        row.addWidget(self.anomaly_combo, 1)
        layout.addLayout(row)

        layout.addWidget(
            hint(
                "Labels are never used to train an autoencoder. The only thing a "
                "class name is used for here is deciding which images to withhold."
            )
        )
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

        # The latent slider is right above this, so the number it has to beat
        # belongs here too - otherwise you are choosing a waist blind.
        self.floor_label = QLabel("-")
        self.floor_label.setObjectName("Metric")
        self.floor_label.setWordWrap(True)
        self.floor_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.floor_label)

        self.problem_label = QLabel("")
        self.problem_label.setObjectName("Warning")
        self.problem_label.setWordWrap(True)
        self.problem_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(self.problem_label)
        return group

    # ----------------------------------------------------------------- queries

    @property
    def variant(self) -> str:
        return VARIANTS[self.variant_combo.currentIndex()]

    def anomaly_class(self) -> int:
        """Index of the held-out class, or NO_ANOMALY."""
        index = self.anomaly_combo.currentIndex() - 1
        if index < 0 or index >= len(self._class_names):
            return NO_ANOMALY
        return index

    def current_config(self) -> AutoencoderConfig:
        return AutoencoderConfig(
            input_shape=self._input_shape,
            variant=self.variant,
            latent_dim=self.latent_slider.value(),
            stages=self.stages_spin.value(),
            base_filters=self.filters_spin.value(),
            hidden_units=self.units_spin.value(),
            denoise=self.denoise_check.isChecked(),
            noise_std=float(self.noise_spin.value()),
            learning_rate=float(self.lr_spin.value()),
        )

    def problems(self) -> list:
        return validate(self.current_config())

    # ------------------------------------------------------------------ public

    def set_data_shape(self, input_shape: tuple, locked: bool) -> None:
        self._input_shape = tuple(int(v) for v in input_shape)
        height, width, channels = self._input_shape
        origin = "from the built dataset" if locked else "planned from the Images tab"
        self.shape_label.setText(
            f"Images are {height}x{width}x{channels} ({origin}), so one image is "
            f"{height * width * channels:,} numbers."
        )
        self.config_changed.emit()

    def set_baseline(self, value: float) -> None:
        """The error to beat, shown next to the waist that has to beat it."""
        self.floor_label.setText(
            "Score to beat: -"
            if value != value
            else f"Score to beat: {value:.5f}"
        )

    def set_classes(self, names: list) -> None:
        """Populate the hold-out list once a dataset exists."""
        self._class_names = list(names or [])
        current = self.anomaly_combo.currentIndex()
        self.anomaly_combo.blockSignals(True)
        self.anomaly_combo.clear()
        self.anomaly_combo.addItem("nothing - train on every class")
        for name in self._class_names:
            self.anomaly_combo.addItem(name)
        self.anomaly_combo.setCurrentIndex(
            current if 0 <= current <= len(self._class_names) else 0
        )
        self.anomaly_combo.blockSignals(False)
        self.anomaly_combo.setEnabled(len(self._class_names) >= 2)
        self.config_changed.emit()

    def set_running(self, running: bool) -> None:
        for widget in (
            self.variant_combo, self.stages_spin, self.filters_spin,
            self.units_spin, self.latent_slider, self.latent_spin,
            self.denoise_check, self.anomaly_combo, self.lr_spin,
            self.sweep_button,
        ):
            widget.setEnabled(not running)
        self.noise_spin.setEnabled(
            not running and self.denoise_check.isChecked()
        )

    def refresh_summary(self) -> None:
        config = self.current_config()
        latent = config.latent_dim

        self.latent_label.setText(
            f"{config.n_inputs:,} numbers  ->  {latent}  "
            f"({config.compression:.0f}x squeeze)"
        )
        self.params_label.setText(
            f"About {estimate_params(config):,} weights. {config.describe()}"
        )

        problems = validate(config)
        self.problem_label.setText("\n".join(problems))

    # ----------------------------------------------------------------- actions

    def _on_slider(self, value: int) -> None:
        self.latent_spin.blockSignals(True)
        self.latent_spin.setValue(value)
        self.latent_spin.blockSignals(False)
        self.config_changed.emit()

    def _on_spin(self, value: int) -> None:
        self.latent_slider.blockSignals(True)
        self.latent_slider.setValue(value)
        self.latent_slider.blockSignals(False)
        self.config_changed.emit()

    def _on_denoise(self, enabled: bool) -> None:
        self.noise_spin.setEnabled(enabled)
        self.config_changed.emit()

    def _sync_variant(self, index: int) -> None:
        self.width_stack.setCurrentIndex(index)
        self.config_changed.emit()

    def set_latent(self, value: int) -> None:
        """Used by the presets, and by the sweep to restore what it changed."""
        self.latent_slider.setValue(int(max(LATENT_MIN, min(LATENT_MAX, value))))
