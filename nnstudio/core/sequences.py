"""Sequence data and recurrent models, free of any UI framework.

The question this workspace exists to answer is not "which recurrent layer is
best". It is the one nobody asks before reaching for an LSTM:

    does the ORDER of this data carry any information at all?

Two things here are built to answer it honestly. The first is a floor, as in
every other workspace: for forecasting it is persistence - answer that the next
value equals the last one you saw - and for classification it is the commonest
class. The second is the shuffle control: take the same data, permute every
sequence's timesteps, and train again. If the score does not move, order was
never the signal and the memory in the model was decoration.

Four tasks ship with it, and they are chosen so those two measurements disagree
in instructive ways. On a random walk persistence is not merely hard to beat -
it is provably optimal, and a network that appears to beat it on validation is
reading noise. On the counting task, shuffling changes nothing, because the
answer never depended on order. Both are traps, deliberately, because a student
who has only ever seen sequences where recurrence helps has not learned when to
use it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_builder import keras_module

# ------------------------------------------------------------------- vocabulary

FORECAST = "forecast"
CLASSIFY = "classify"

SINE = "sine"
WALK = "walk"
ORDER = "order"
COUNT = "count"

TASKS = (SINE, WALK, ORDER, COUNT)

TASK_LABELS = {
    SINE: "Sine wave - predict the next value",
    WALK: "Random walk - predict the next value",
    ORDER: "Which spike came first? - order is the whole signal",
    COUNT: "More ups than downs? - order is irrelevant",
}

TASK_KINDS = {SINE: FORECAST, WALK: FORECAST, ORDER: CLASSIFY, COUNT: CLASSIFY}

# What each task is actually for, said plainly. These are not decoration: two of
# the four exist to be lost, and a student who does not know that reads a bad
# result as their own mistake.
TASK_NOTES = {
    SINE: (
        "A noisy sine wave. The next value genuinely depends on where you are "
        "in the cycle, so a model with memory should beat persistence clearly. "
        "This is the honest case, and the only one of the four where reaching "
        "for a recurrent layer is obviously right."
    ),
    WALK: (
        "Each step is the previous value plus fresh noise. Nothing in the past "
        "predicts the next step beyond the last value itself, so persistence is "
        "not just a strong baseline here - it is mathematically the best any "
        "model can do. Expect to lose. If your network appears to win, it is "
        "reading noise in the validation split, not learning the series."
    ),
    ORDER: (
        "Two spikes, one up and one down, at random positions. The label is "
        "which came first. Both classes contain exactly the same values, so no "
        "amount of counting can separate them - the answer lives entirely in "
        "the arrangement. Note that a Dense net still solves this easily: with "
        "a fixed window it gets one input per timestep and can simply compare "
        "two of them. It is the shuffle probe, not the sweep, that shows the "
        "answer really was about order."
    ),
    COUNT: (
        "Is the sum above zero? The answer does not depend on the order of the "
        "steps at all. Shuffle the timesteps and a correct model should score "
        "exactly the same. If yours drops, it memorised positions instead of "
        "learning the rule."
    ),
}

LSTM = "lstm"
GRU = "gru"
RNN = "rnn"
CONV1D = "conv1d"
DENSE = "dense"

KINDS = (LSTM, GRU, RNN, CONV1D, DENSE)

KIND_LABELS = {
    LSTM: "LSTM - gated memory, the usual default",
    GRU: "GRU - the same idea with fewer gates",
    RNN: "Simple RNN - one state, no gates",
    CONV1D: "Conv1D - local patterns, shared across time",
    DENSE: "Dense on the flattened window - one weight per timestep",
}

# The dense arm matters more than it looks, and it is easy to describe wrongly.
# It is NOT order-blind: flattening a fixed-length window gives it one input per
# timestep, so it can perfectly well learn "the value at step 3 is larger than
# the value at step 17". Measured on this project's own order task it reached
# 0.99 accuracy doing exactly that.
#
# What it genuinely cannot do is share what it learns between positions. A
# pattern it has learned at step 3 means nothing to it at step 4, it needs a
# fixed window length forever, and it has no way to generalise to a longer one.
# That is the real trade, and it is why the dense arm is the baseline every
# recurrent layer has to beat before it has earned its cost.
KIND_NOTES = {
    LSTM: "Three gates decide what to keep, what to forget and what to emit.",
    GRU: "Two gates instead of three. Usually as good, always cheaper.",
    RNN: "No gates, so the gradient fades over long sequences. Keep it short.",
    CONV1D: "Slides one small kernel over every position. Sees local shape, and "
            "reuses what it learns at every step.",
    DENSE: "One weight per timestep, none of them shared. Strong on a short "
           "fixed window, useless the moment the length changes.",
}

CONV_KERNEL = 5
MIN_LENGTH = 4
MAX_LENGTH = 200


class SequenceError(ValueError):
    """Raised when a configuration cannot describe a working model."""


# ------------------------------------------------------------------------ data

@dataclass
class SequenceSpec:
    """How to generate one synthetic sequence dataset."""

    task: str = SINE
    n_sequences: int = 1200
    length: int = 24
    noise: float = 0.05
    val_fraction: float = 0.25
    seed: int = 7
    shuffle_time: bool = False


@dataclass
class SequenceBundle:
    """Windows in, answers out, already split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    name: str
    task: str
    kind: str
    length: int
    n_features: int
    class_names: tuple = ()
    shuffled: bool = False
    series: np.ndarray | None = None      # the source series, for plotting

    @property
    def n_outputs(self) -> int:
        return len(self.class_names) if self.kind == CLASSIFY else 1

    def describe(self) -> str:
        head = (
            f"{self.name}: {len(self.x_train)} training / {len(self.x_val)} "
            f"validation windows of {self.length} steps"
        )
        if self.kind == CLASSIFY:
            head += f", {len(self.class_names)} classes ({', '.join(self.class_names)})"
        else:
            head += ", predicting the next value"
        if self.shuffled:
            head += ". TIMESTEPS SHUFFLED - order has been destroyed on purpose"
        return head + "."


def shuffle_timesteps(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the timesteps of every sequence, each with its own permutation.

    A single shared permutation would only relabel time, and any model able to
    learn the original could learn the relabelling just as easily. Permuting
    each sequence separately destroys order while leaving the multiset of values
    in every window exactly as it was - so anything still learnable afterwards
    was never about order in the first place.
    """
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[i][rng.permutation(x.shape[1])]
    return out


def _windows(series: np.ndarray, length: int) -> tuple:
    """Slide a window over one long series. Target is the step after it."""
    n = len(series) - length
    x = np.empty((n, length, 1), dtype="float32")
    y = np.empty((n, 1), dtype="float32")
    for i in range(n):
        x[i, :, 0] = series[i:i + length]
        y[i, 0] = series[i + length]
    return x, y


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype="float32")
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def _sine_series(spec: SequenceSpec, rng) -> np.ndarray:
    total = spec.n_sequences + spec.length
    period = max(6.0, spec.length / 2.0)
    t = np.arange(total, dtype="float64")
    clean = np.sin(2 * np.pi * t / period)
    return (clean + rng.normal(0.0, spec.noise, total)).astype("float32")


def _walk_series(spec: SequenceSpec, rng) -> np.ndarray:
    total = spec.n_sequences + spec.length
    steps = rng.normal(0.0, max(spec.noise, 1e-4), total)
    return np.cumsum(steps).astype("float32")


def _order_set(spec: SequenceSpec, rng) -> tuple:
    """Two spikes at random positions; the label is which one came first."""
    n, length = spec.n_sequences, spec.length
    x = rng.normal(0.0, spec.noise, (n, length, 1)).astype("float32")
    labels = np.empty(n, dtype="int64")
    for i in range(n):
        a, b = rng.choice(length, size=2, replace=False)
        x[i, a, 0] += 2.0
        x[i, b, 0] -= 2.0
        labels[i] = 1 if a < b else 0
    return x, labels, ("down first", "up first")


def _count_set(spec: SequenceSpec, rng) -> tuple:
    """Steps of plus or minus one; the label is whether the sum is positive."""
    n, length = spec.n_sequences, spec.length
    signs = rng.choice((-1.0, 1.0), size=(n, length))
    totals = signs.sum(axis=1)
    # A tie has no honest answer, so nudge one step and keep the label clean.
    for i in np.where(totals == 0)[0]:
        signs[i, 0] = 1.0
        totals[i] = signs[i].sum()
    x = (signs + rng.normal(0.0, spec.noise, (n, length))).astype("float32")
    return x[..., None], (totals > 0).astype("int64"), ("more down", "more up")


def make_sequences(spec: SequenceSpec, on_progress=None, should_stop=None) -> SequenceBundle:
    """Generate one of the four datasets, already split.

    Forecast tasks are split CHRONOLOGICALLY - validation is the tail of the
    series, never a random sample of it. A random split would put windows from
    after the validation point into training, which is the future leaking into
    the past, and it makes every forecast look better than it is.
    """
    task = str(spec.task).lower()
    if task not in TASKS:
        raise SequenceError(f"Unknown task '{spec.task}'")
    length = int(spec.length)
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        raise SequenceError(
            f"Sequence length must be between {MIN_LENGTH} and {MAX_LENGTH}"
        )

    rng = np.random.default_rng(int(spec.seed))
    if on_progress:
        on_progress(0, 100, f"Generating {TASK_LABELS[task]}", 0.0)
    if should_stop and should_stop():
        from .vision import BuildCancelled

        raise BuildCancelled()

    kind = TASK_KINDS[task]
    series = None
    class_names: tuple = ()

    if kind == FORECAST:
        series = _sine_series(spec, rng) if task == SINE else _walk_series(spec, rng)
        x, y = _windows(series, length)
        cut = int(len(x) * (1.0 - float(spec.val_fraction)))
        x_train, y_train, x_val, y_val = x[:cut], y[:cut], x[cut:], y[cut:]
    else:
        x, labels, class_names = (
            _order_set(spec, rng) if task == ORDER else _count_set(spec, rng)
        )
        order = rng.permutation(len(x))
        x, labels = x[order], labels[order]
        cut = int(len(x) * (1.0 - float(spec.val_fraction)))
        y = _one_hot(labels, len(class_names))
        x_train, y_train, x_val, y_val = x[:cut], y[:cut], x[cut:], y[cut:]

    if spec.shuffle_time:
        control = np.random.default_rng(int(spec.seed) + 1)
        x_train = shuffle_timesteps(x_train, control)
        x_val = shuffle_timesteps(x_val, control)

    if on_progress:
        on_progress(100, 100, "Done", 0.0)

    return SequenceBundle(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
        name=TASK_LABELS[task], task=task, kind=kind, length=length,
        n_features=1, class_names=class_names,
        shuffled=bool(spec.shuffle_time), series=series,
    )


# -------------------------------------------------------------------- baselines

def persistence_baseline(bundle: SequenceBundle) -> float:
    """MSE of answering 'the next value is the last one you saw'.

    The forecasting floor. It costs nothing, needs no training, and on a random
    walk it cannot be beaten by anything.
    """
    if bundle.kind != FORECAST:
        raise SequenceError("Persistence only applies to a forecasting task")
    last = bundle.x_val[:, -1, 0]
    difference = bundle.y_val.ravel() - last
    return float(np.mean(difference * difference))


def majority_baseline(bundle: SequenceBundle) -> float:
    """Accuracy of always answering with the commonest class in training."""
    if bundle.kind != CLASSIFY:
        raise SequenceError("A majority class only applies to classification")
    train = np.argmax(bundle.y_train, axis=1)
    winner = int(np.bincount(train, minlength=len(bundle.class_names)).argmax())
    return float(np.mean(np.argmax(bundle.y_val, axis=1) == winner))


def baseline(bundle: SequenceBundle) -> dict:
    """The give-up score for this bundle, and what it means.

    `better_is_lower` exists because the two tasks disagree about which
    direction is good, and every readout that compares against this floor has to
    know which way round it is.
    """
    if bundle.kind == FORECAST:
        return {
            "value": persistence_baseline(bundle),
            "metric": "mse",
            "name": "persistence",
            "better_is_lower": True,
            "explain": (
                "That is the error you get by answering that the next value "
                "equals the last one in the window - no model, no training. "
                "Until the validation MSE drops below it, the network has "
                "learned nothing worth having."
            ),
        }
    return {
        "value": majority_baseline(bundle),
        "metric": "accuracy",
        "name": "majority class",
        "better_is_lower": False,
        "explain": (
            "That is the accuracy of always answering with the commonest class "
            "in the training set - no model, no training. Until the validation "
            "accuracy rises above it, the network has learned nothing."
        ),
    }


# ------------------------------------------------------------------ the model

@dataclass
class SequenceConfig:
    """One sequence model, sized for CPU training."""

    length: int = 24
    n_features: int = 1
    kind: str = LSTM
    units: int = 32
    layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    task_kind: str = FORECAST
    n_outputs: int = 1
    learning_rate: float = 0.005
    optimizer: str = "adam"

    @property
    def reads_order(self) -> bool:
        """Whether this architecture can tell 'before' from 'after' at all."""
        return self.kind in (LSTM, GRU, RNN)

    @property
    def n_inputs(self) -> int:
        return int(self.length) * int(self.n_features)

    def stack(self) -> list:
        """Layer descriptors for the diagram."""
        columns = [
            {
                "name": "Window",
                "detail": f"{self.length} steps x {self.n_features}",
                "kind": "input",
                "values": self.n_inputs,
            }
        ]
        width = int(self.units) * (2 if self.bidirectional else 1)
        for layer in range(1, int(self.layers) + 1):
            last = layer == int(self.layers)
            if self.kind in (LSTM, GRU, RNN):
                detail = f"{self.units} units"
                if self.bidirectional:
                    detail += " x2 (both directions)"
                detail += ", one step at a time" if not last else ", final state"
                values = width * (int(self.length) if not last else 1)
            elif self.kind == CONV1D:
                detail = f"{self.units} filters, kernel {CONV_KERNEL}"
                values = int(self.units) * (int(self.length) if not last else 1)
            else:
                detail = f"{self.units} units"
                values = int(self.units)
            columns.append({
                "name": f"{self.kind.upper()} {layer}",
                "detail": detail,
                "kind": "hidden",
                "values": values,
            })
        columns.append({
            "name": "Answer",
            "detail": ("next value" if self.task_kind == FORECAST
                       else f"{self.n_outputs} classes"),
            "kind": "output",
            "values": int(self.n_outputs),
        })
        return columns

    def describe(self) -> str:
        head = f"{KIND_LABELS.get(self.kind, self.kind)}, {self.units} units"
        if self.layers > 1:
            head += f" x {self.layers} layers"
        if self.bidirectional and self.reads_order:
            head += ", bidirectional"
        return head


def validate(config: SequenceConfig) -> list:
    problems = []
    if config.kind not in KINDS:
        problems.append(f"Unknown architecture '{config.kind}'")
    if int(config.units) < 1:
        problems.append("A layer needs at least one unit")
    if int(config.layers) < 1:
        problems.append("At least one layer is needed")
    if config.kind == CONV1D and int(config.length) < CONV_KERNEL:
        problems.append(
            f"Conv1D uses a kernel of {CONV_KERNEL}, so the window must be at "
            f"least {CONV_KERNEL} steps long - this one is {config.length}"
        )
    if config.bidirectional and not config.reads_order:
        problems.append(
            f"{config.kind} has no direction to reverse, so bidirectional means "
            "nothing here. Turn it off, or pick a recurrent layer"
        )
    if config.task_kind == CLASSIFY and int(config.n_outputs) < 2:
        problems.append("A classification head needs at least two classes")
    return problems


def estimate_params(config: SequenceConfig) -> int:
    """Roughly how many weights this will have, before Keras is loaded."""
    features, units = int(config.n_features), int(config.units)
    directions = 2 if (config.bidirectional and config.reads_order) else 1
    total, incoming = 0, features

    for layer in range(int(config.layers)):
        if config.kind in (LSTM, GRU, RNN):
            gates = {LSTM: 4, GRU: 3, RNN: 1}[config.kind]
            per = gates * ((incoming + units) * units + units)
            if config.kind == GRU:
                per += 3 * units          # reset_after keeps a second bias set
            total += per * directions
            incoming = units * directions
        elif config.kind == CONV1D:
            total += CONV_KERNEL * incoming * units + units
            incoming = units
        else:
            flat = incoming * int(config.length) if layer == 0 else incoming
            total += flat * units + units
            incoming = units

    head_in = incoming if config.kind != DENSE else incoming
    total += head_in * int(config.n_outputs) + int(config.n_outputs)
    return int(total)


def _optimizer(keras, name: str, learning_rate: float):
    factories = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
        "nadam": keras.optimizers.Nadam,
    }
    factory = factories.get(str(name).lower(), keras.optimizers.Adam)
    return factory(learning_rate=float(learning_rate))


def build_sequence_model(config: SequenceConfig):
    """Compile one sequence model. Raises SequenceError on a bad configuration."""
    problems = validate(config)
    if problems:
        raise SequenceError("; ".join(problems))

    keras = keras_module()
    layers = keras.layers

    inputs = keras.Input(shape=(int(config.length), int(config.n_features)),
                         name="window")
    x = inputs
    total = int(config.layers)

    for index in range(total):
        last = index == total - 1
        if config.kind in (LSTM, GRU, RNN):
            factory = {LSTM: layers.LSTM, GRU: layers.GRU,
                       RNN: layers.SimpleRNN}[config.kind]
            # Only the last recurrent layer collapses time; the ones before it
            # must hand the whole sequence on, or there is nothing left to read.
            cell = factory(int(config.units), return_sequences=not last,
                           name=f"{config.kind}_{index + 1}")
            x = (layers.Bidirectional(cell, name=f"bi_{index + 1}")
                 if config.bidirectional else cell)(x)
        elif config.kind == CONV1D:
            x = layers.Conv1D(int(config.units), CONV_KERNEL, padding="same",
                              activation="relu", name=f"conv_{index + 1}")(x)
            if last:
                x = layers.GlobalMaxPooling1D(name="pool")(x)
        else:
            if index == 0:
                x = layers.Flatten(name="flatten")(x)
            x = layers.Dense(int(config.units), activation="relu",
                             name=f"dense_{index + 1}")(x)
        if config.dropout > 0:
            x = layers.Dropout(float(config.dropout), name=f"drop_{index + 1}")(x)

    if config.task_kind == CLASSIFY:
        outputs = layers.Dense(int(config.n_outputs), activation="softmax",
                               name="answer")(x)
        loss, metrics = "categorical_crossentropy", ["accuracy"]
    else:
        outputs = layers.Dense(1, name="answer")(x)
        loss, metrics = "mse", ["mae"]

    model = keras.Model(inputs, outputs, name=f"{config.kind}_sequence")
    model.compile(
        optimizer=_optimizer(keras, config.optimizer, config.learning_rate),
        loss=loss, metrics=metrics,
    )
    return model


def count_params(model) -> tuple:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return trainable, total


# ------------------------------------------------------------------- inference

def predict(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x, verbose=0))


def forecast_rows(model, bundle: SequenceBundle, count: int = 120) -> dict:
    """Actual, predicted and persistence over the first stretch of validation.

    All three on the same axis, because a forecast chart that shows only the
    prediction is the most convincing lie in this whole subject: a curve that
    lags the truth by one step looks like a near-perfect fit.
    """
    if bundle.kind != FORECAST:
        raise SequenceError("Only a forecasting task has a series to plot")
    take = min(int(count), len(bundle.x_val))
    x = bundle.x_val[:take]
    return {
        "actual": bundle.y_val[:take].ravel().astype("float32"),
        "predicted": predict(model, x).ravel().astype("float32"),
        "persistence": x[:, -1, 0].astype("float32"),
    }


def confusion(model, bundle: SequenceBundle) -> np.ndarray:
    """Rows are the true class, columns what the model answered."""
    if bundle.kind != CLASSIFY:
        raise SequenceError("Only a classification task has a confusion matrix")
    n = len(bundle.class_names)
    truth = np.argmax(bundle.y_val, axis=1)
    guess = np.argmax(predict(model, bundle.x_val), axis=1)
    table = np.zeros((n, n), dtype="int64")
    for t, g in zip(truth, guess):
        table[t, g] += 1
    return table
