"""Declarative network configuration compiled into a Keras model.

TensorFlow is imported lazily so the UI starts instantly; the first import
only happens when a model is actually built, inside the training thread.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .dataset import TASK_BINARY, TASK_MULTICLASS, TASK_REGRESSION

HIDDEN_ACTIVATIONS = (
    "relu",
    "tanh",
    "sigmoid",
    "elu",
    "selu",
    "gelu",
    "softplus",
    "swish",
    "linear",
)

OPTIMIZERS = ("adam", "sgd", "rmsprop", "nadam", "adamw")


@dataclass(frozen=True)
class OutputMode:
    """One choice of output layer: activation + the loss that belongs with it."""

    key: str
    label: str
    activation: str
    loss: str
    metrics: tuple
    task: str
    hint: str


OUTPUT_MODES = {
    "softmax": OutputMode(
        key="softmax",
        label="Softmax - multiclass probabilities",
        activation="softmax",
        loss="categorical_crossentropy",
        metrics=("accuracy",),
        task=TASK_MULTICLASS,
        hint=(
            "One neuron per class. Outputs sum to 1, so they read as probabilities. "
            "Paired with categorical crossentropy."
        ),
    ),
    "sigmoid": OutputMode(
        key="sigmoid",
        label="Sigmoid - binary decision",
        activation="sigmoid",
        loss="binary_crossentropy",
        metrics=("accuracy",),
        task=TASK_BINARY,
        hint=(
            "A single neuron squashed into 0..1. Above 0.5 is the positive class. "
            "Paired with binary crossentropy."
        ),
    ),
    "linear": OutputMode(
        key="linear",
        label="Linear - regression",
        activation="linear",
        loss="mse",
        metrics=("mae",),
        task=TASK_REGRESSION,
        hint=(
            "No activation at all: the neuron emits any real number. "
            "Paired with mean squared error."
        ),
    ),
    "relu": OutputMode(
        key="relu",
        label="ReLU - non-negative regression",
        activation="relu",
        loss="mse",
        metrics=("mae",),
        task=TASK_REGRESSION,
        hint=(
            "Clamps negatives to exactly 0. Only for targets that cannot be negative "
            "(counts, prices, durations). Dead neurons stay dead."
        ),
    ),
    "softplus": OutputMode(
        key="softplus",
        label="Softplus - strictly positive regression",
        activation="softplus",
        loss="mse",
        metrics=("mae",),
        task=TASK_REGRESSION,
        hint=(
            "A smooth ReLU: always > 0 and always differentiable, so it never dies. "
            "The safer choice when the target must stay positive."
        ),
    ),
}

MODES_BY_TASK = {
    TASK_BINARY: ("sigmoid",),
    TASK_MULTICLASS: ("softmax",),
    TASK_REGRESSION: ("linear", "softplus", "relu"),
}


@dataclass
class LayerSpec:
    """A single hidden layer."""

    units: int = 16
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False


@dataclass
class NetworkConfig:
    """The whole architecture, as data. Nothing here depends on Keras."""

    n_inputs: int = 4
    n_outputs: int = 3
    hidden: list = field(default_factory=list)
    output_mode: str = "softmax"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    l2: float = 0.0

    @property
    def mode(self) -> OutputMode:
        return OUTPUT_MODES[self.output_mode]

    def layer_summary(self) -> list:
        """Column descriptors used by the network diagram."""
        columns = [
            {
                "name": "Input",
                "units": self.n_inputs,
                "activation": "-",
                "kind": "input",
            }
        ]
        for i, layer in enumerate(self.hidden, start=1):
            extras = []
            if layer.batch_norm:
                extras.append("BN")
            if layer.dropout > 0:
                extras.append(f"drop {layer.dropout:.2f}")
            columns.append(
                {
                    "name": f"Hidden {i}",
                    "units": layer.units,
                    "activation": layer.activation,
                    "kind": "hidden",
                    "extras": ", ".join(extras),
                }
            )
        columns.append(
            {
                "name": "Output",
                "units": self.n_outputs,
                "activation": self.mode.activation,
                "kind": "output",
            }
        )
        return columns


def estimate_params(config: NetworkConfig) -> int:
    """Trainable parameter count, computed without touching TensorFlow."""
    total = 0
    previous = config.n_inputs
    for layer in config.hidden:
        total += previous * layer.units + layer.units
        if layer.batch_norm:
            total += 4 * layer.units
        previous = layer.units
    total += previous * config.n_outputs + config.n_outputs
    return total


def validate(config: NetworkConfig) -> list:
    """Return human-readable warnings; an empty list means nothing to flag."""
    warnings = []
    mode = config.mode
    if mode.key == "softmax" and config.n_outputs < 2:
        warnings.append("Softmax needs at least 2 output neurons.")
    if mode.key == "sigmoid" and config.n_outputs != 1:
        warnings.append("A binary sigmoid output should have exactly 1 neuron.")
    if mode.task == TASK_REGRESSION and config.n_outputs != 1:
        warnings.append("Regression normally uses a single output neuron.")
    if not config.hidden:
        warnings.append(
            "No hidden layers: this is plain linear/logistic regression, "
            "it cannot learn non-linear boundaries."
        )
    if estimate_params(config) > 5_000_000:
        warnings.append("Over 5M parameters - training on CPU will be slow.")
    for i, layer in enumerate(config.hidden, start=1):
        if layer.dropout >= 0.6:
            warnings.append(f"Hidden {i}: dropout {layer.dropout:.2f} is very aggressive.")
    return warnings


def keras_module():
    """Import Keras once, quietly."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    from tensorflow import keras  # noqa: PLC0415 - deliberately lazy

    return keras


def _build_optimizer(keras, config: NetworkConfig):
    name = config.optimizer.lower()
    lr = float(config.learning_rate)
    factories = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=lr),
        "sgd": lambda: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate=lr),
        "nadam": lambda: keras.optimizers.Nadam(learning_rate=lr),
    }
    if name == "adamw":
        try:
            return keras.optimizers.AdamW(learning_rate=lr)
        except AttributeError:
            return keras.optimizers.Adam(learning_rate=lr)
    return factories.get(name, factories["adam"])()


def build_model(config: NetworkConfig):
    """Compile the configuration into a Sequential Keras model."""
    keras = keras_module()
    regularizer = keras.regularizers.l2(config.l2) if config.l2 > 0 else None

    model = keras.Sequential(name="interactive_network")
    model.add(keras.Input(shape=(config.n_inputs,), name="input"))

    for i, layer in enumerate(config.hidden, start=1):
        # With batch norm the activation moves after the normalisation, which
        # is the ordering that actually helps training.
        model.add(
            keras.layers.Dense(
                layer.units,
                activation=None if layer.batch_norm else layer.activation,
                kernel_regularizer=regularizer,
                name=f"dense_{i}",
            )
        )
        if layer.batch_norm:
            model.add(keras.layers.BatchNormalization(name=f"batchnorm_{i}"))
            model.add(keras.layers.Activation(layer.activation, name=f"activation_{i}"))
        if layer.dropout > 0:
            model.add(keras.layers.Dropout(layer.dropout, name=f"dropout_{i}"))

    mode = config.mode
    model.add(
        keras.layers.Dense(config.n_outputs, activation=mode.activation, name="output")
    )
    model.compile(
        optimizer=_build_optimizer(keras, config),
        loss=mode.loss,
        metrics=list(mode.metrics),
    )
    return model


def summary_text(model) -> str:
    lines = []
    model.summary(print_fn=lines.append, line_length=88)
    return "\n".join(lines)
