"""Convolutional models: a small ResNet built by hand, and transfer learning.

The residual block is written out explicitly rather than pulled from a library,
because the whole point of the tab is watching what the skip connection does.
Turning `use_skip` off leaves the exact same layers, same depth, same parameter
count - only the addition is gone. That is the controlled experiment that made
ResNet famous: plain deep networks were not overfitting, they were failing to
optimise at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_builder import keras_module

BACKBONES = ("ResNet50", "ResNet50V2")

MIN_TRANSFER_SIZE = 32


@dataclass
class ResNetConfig:
    """A ResNet-style stack sized for CPU training."""

    input_shape: tuple = (32, 32, 3)
    n_classes: int = 3
    stem_filters: int = 16
    stages: int = 3
    blocks_per_stage: int = 1
    use_skip: bool = True
    dropout: float = 0.0
    augment: bool = True
    learning_rate: float = 0.001
    optimizer: str = "adam"

    @property
    def depth(self) -> int:
        """Weight layers on the main path: stem + 2 per block + classifier."""
        return 1 + 2 * self.stages * self.blocks_per_stage + 1

    def stage_filters(self) -> list:
        return [self.stem_filters * (2 ** i) for i in range(self.stages)]

    def summary_columns(self) -> list:
        """Column descriptors for the diagram widget."""
        h, w, c = self.input_shape
        columns = [
            {"name": "Input", "detail": f"{h}x{w}x{c}", "kind": "input"}
        ]
        size = h
        for stage, filters in enumerate(self.stage_filters(), start=1):
            if stage > 1:
                size = max(1, size // 2)
            columns.append(
                {
                    "name": f"Stage {stage}",
                    "detail": f"{self.blocks_per_stage} x block, {filters}f, {size}x{size}",
                    "kind": "hidden",
                    "skip": self.use_skip,
                }
            )
        columns.append(
            {"name": "GAP + Dense", "detail": f"{self.n_classes} classes", "kind": "output"}
        )
        return columns


@dataclass
class TransferConfig:
    """A frozen ImageNet backbone with a small trainable head."""

    input_shape: tuple = (64, 64, 3)
    n_classes: int = 3
    backbone: str = "ResNet50"
    freeze_backbone: bool = True
    head_units: int = 64
    dropout: float = 0.2
    augment: bool = False
    learning_rate: float = 0.001
    optimizer: str = "adam"


def _optimizer(keras, name: str, learning_rate: float):
    factories = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
        "nadam": keras.optimizers.Nadam,
    }
    factory = factories.get(name.lower(), keras.optimizers.Adam)
    return factory(learning_rate=float(learning_rate))


def _augmentation(keras):
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomTranslation(0.12, 0.12),
            keras.layers.RandomRotation(0.08),
        ],
        name="augmentation",
    )


def residual_block(keras, x, filters: int, stride: int, use_skip: bool, tag: str):
    """Conv-BN-ReLU twice, then add the input back in (or do not).

    Both branches keep identical layers and parameter counts. `use_skip=False`
    simply drops the addition, which is exactly the ablation to look at.
    """
    shortcut = x

    y = keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        name=f"{tag}_conv1",
    )(x)
    y = keras.layers.BatchNormalization(name=f"{tag}_bn1")(y)
    y = keras.layers.ReLU(name=f"{tag}_relu1")(y)

    y = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{tag}_conv2"
    )(y)
    y = keras.layers.BatchNormalization(name=f"{tag}_bn2")(y)

    if use_skip:
        # The shortcut only needs its own conv when shape or width changed.
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(
                filters, 1, strides=stride, use_bias=False, name=f"{tag}_proj"
            )(shortcut)
            shortcut = keras.layers.BatchNormalization(name=f"{tag}_projbn")(shortcut)
        y = keras.layers.Add(name=f"{tag}_add")([y, shortcut])

    return keras.layers.ReLU(name=f"{tag}_out")(y)


def build_resnet(config: ResNetConfig):
    """Compile the hand-built stack into a Keras model."""
    keras = keras_module()

    inputs = keras.Input(shape=config.input_shape, name="image")
    x = inputs
    if config.augment:
        x = _augmentation(keras)(x)
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    x = keras.layers.Conv2D(
        config.stem_filters, 3, padding="same", use_bias=False, name="stem_conv"
    )(x)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    for stage, filters in enumerate(config.stage_filters(), start=1):
        for block in range(1, config.blocks_per_stage + 1):
            stride = 2 if (stage > 1 and block == 1) else 1
            x = residual_block(
                keras, x, filters, stride, config.use_skip, f"s{stage}b{block}"
            )

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    if config.dropout > 0:
        x = keras.layers.Dropout(config.dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        config.n_classes, activation="softmax", name="predictions"
    )(x)

    name = "resnet" if config.use_skip else "plain_cnn"
    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=_optimizer(keras, config.optimizer, config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer(config: TransferConfig):
    """A pretrained ImageNet backbone plus a small trainable head.

    The first call downloads roughly 98 MB of weights into ~/.keras/models.
    """
    keras = keras_module()

    height, width, channels = config.input_shape
    if min(height, width) < MIN_TRANSFER_SIZE:
        raise ValueError(
            f"{config.backbone} needs images of at least "
            f"{MIN_TRANSFER_SIZE}x{MIN_TRANSFER_SIZE}"
        )
    if channels != 3:
        raise ValueError(f"{config.backbone} expects 3-channel RGB images")

    factory = getattr(keras.applications, config.backbone)
    preprocess = (
        keras.applications.resnet_v2.preprocess_input
        if config.backbone.endswith("V2")
        else keras.applications.resnet.preprocess_input
    )

    backbone = factory(
        include_top=False,
        weights="imagenet",
        input_shape=config.input_shape,
        pooling="avg",
    )
    backbone.trainable = not config.freeze_backbone

    inputs = keras.Input(shape=config.input_shape, name="image")
    x = inputs
    if config.augment:
        x = _augmentation(keras)(x)
    x = keras.layers.Lambda(preprocess, name="imagenet_preprocess")(x)
    # A frozen backbone must also stay in inference mode, or its batch-norm
    # statistics drift while "frozen" and the head learns against a moving target.
    x = backbone(x, training=not config.freeze_backbone)

    if config.head_units > 0:
        x = keras.layers.Dense(config.head_units, activation="relu", name="head")(x)
    if config.dropout > 0:
        x = keras.layers.Dropout(config.dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        config.n_classes, activation="softmax", name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone.lower()}_transfer")
    model.compile(
        optimizer=_optimizer(keras, config.optimizer, config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def weights_are_cached(backbone: str) -> bool:
    """True when the ImageNet weights are already on disk."""
    from pathlib import Path  # noqa: PLC0415

    models = Path.home() / ".keras" / "models"
    if not models.is_dir():
        return False
    stem = backbone.lower().replace("v2", "v2")
    return any(
        stem in p.name.lower() and "notop" in p.name.lower() for p in models.iterdir()
    )


def count_params(model) -> tuple:
    """(trainable, total) parameter counts.

    The gap between the two is the whole story of transfer learning: a frozen
    backbone contributes 23M total and 0 trainable.
    """
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(v.shape)) for v in model.trainable_weights))
    return trainable, total
