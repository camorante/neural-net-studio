"""A tiny transformer with its attention map exposed, free of any UI framework.

Two measurements shape this whole workspace, and both exist because the obvious
version of an attention demo teaches something false.

The first is **positional encoding**. Self-attention is permutation-invariant:
strip the position information and the model literally cannot tell "first" from
"last", no matter how long you train it. That is not a tuning problem, it is
what the mechanism is. `TransformerConfig.positional` turns it off so the
failure can be watched rather than asserted.

The second is **faithfulness**. An attention heatmap looks like the model
telling you where it looked, and it is almost irresistible to read it that way.
But the weights are not a causal account of anything. So this module ships
`faithfulness()`, which replaces the token the model attended to most and
measures what actually happens to the answer - against the control of replacing
some other token instead. A map that survives that test earned its reading. A
map that does not is decoration, and the workspace says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_builder import keras_module

# ------------------------------------------------------------------ vocabulary

# Token 0 is the cue marker in the `match` task and never appears as content, so
# it is reserved everywhere. Keeping it reserved in the other tasks too means a
# vocabulary size means the same thing whichever task is loaded.
CUE = 0

FIRST = "first"
MATCH = "match"
MAJORITY = "majority"

TASKS = (FIRST, MATCH, MAJORITY)

TASK_LABELS = {
    FIRST: "First token - what did the sequence start with?",
    MATCH: "After the cue - which token follows the marker?",
    MAJORITY: "Most common token - which appears most often?",
}

# What each task is for, and what its attention map should look like if the
# model solved it the way you would. These are predictions a student can check
# by eye, which is the whole reason the tasks are shaped like this.
TASK_NOTES = {
    FIRST: (
        "The answer is whatever token sits at position 0. Attention should "
        "collapse onto that one column and ignore everything else. Without "
        "positional encoding 'first' is not a property the model can perceive "
        "at all, so it falls back to guessing the commonest token in the "
        "sequence - which is right about a quarter of the time, only because "
        "the first token is itself one of the ones being counted."
    ),
    MATCH: (
        "A cue marker appears once, somewhere. The answer is the token directly "
        "after it. Attention should spike on a single column that moves with "
        "the cue - a different column for every sequence. It needs TWO blocks: "
        "one to work out where the cue is relative to each position, one to "
        "fetch the token after it. A single block never got past 31%."
    ),
    MAJORITY: (
        "The answer is whichever token appears most often. Order is irrelevant, "
        "so this one works perfectly well WITHOUT positional encoding - and its "
        "attention map should be diffuse, because there is no single position "
        "worth looking at. A sharp map here would be suspicious."
    ),
}

TASK_NEEDS_POSITIONS = {FIRST: True, MATCH: True, MAJORITY: False}

MEAN = "mean"
FIRST_TOKEN = "first"
POOLS = (MEAN, FIRST_TOKEN)

POOL_LABELS = {
    MEAN: "Average every position",
    FIRST_TOKEN: "Read position 0 only",
}

MIN_LENGTH = 4
MAX_LENGTH = 64
MIN_VOCAB = 3
MAX_VOCAB = 32


class TransformerError(ValueError):
    """Raised when a configuration cannot describe a working transformer."""


# ------------------------------------------------------------------------ data

@dataclass
class TokenSpec:
    """How to generate one synthetic token dataset."""

    task: str = MATCH
    n_sequences: int = 1600
    length: int = 12
    vocab: int = 8
    val_fraction: float = 0.25
    seed: int = 7


@dataclass
class TokenBundle:
    """Integer tokens in, a class out, already split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    name: str
    task: str
    length: int
    vocab: int
    class_names: tuple

    @property
    def n_outputs(self) -> int:
        return len(self.class_names)

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.x_train)} training / {len(self.x_val)} "
            f"validation sequences of {self.length} tokens, vocabulary of "
            f"{self.vocab}, {self.n_outputs} possible answers."
        )


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype="float32")
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def make_tokens(spec: TokenSpec, on_progress=None, should_stop=None) -> TokenBundle:
    """Generate one of the three token tasks, already split.

    Content tokens run from 1 to vocab-1; token 0 is the cue and never appears
    as content. The label is always a content token, so the class names are the
    same list for all three tasks and a model can be compared across them.
    """
    task = str(spec.task).lower()
    if task not in TASKS:
        raise TransformerError(f"Unknown task '{spec.task}'")
    length, vocab = int(spec.length), int(spec.vocab)
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        raise TransformerError(
            f"Sequence length must be between {MIN_LENGTH} and {MAX_LENGTH}")
    if not MIN_VOCAB <= vocab <= MAX_VOCAB:
        raise TransformerError(
            f"Vocabulary must be between {MIN_VOCAB} and {MAX_VOCAB}")

    rng = np.random.default_rng(int(spec.seed))
    n = int(spec.n_sequences)
    if on_progress:
        on_progress(0, 100, f"Generating {TASK_LABELS[task]}", 0.0)
    if should_stop and should_stop():
        from .vision import BuildCancelled

        raise BuildCancelled()

    content = np.arange(1, vocab)
    x = rng.choice(content, size=(n, length)).astype("int32")

    if task == FIRST:
        labels = x[:, 0].copy()
    elif task == MATCH:
        # The cue never lands on the last position, or there would be nothing
        # after it to point at.
        where = rng.integers(0, length - 1, size=n)
        rows = np.arange(n)
        x[rows, where] = CUE
        labels = x[rows, where + 1].copy()
    else:
        # Force a clear winner, so every sequence has one honest answer.
        labels = np.empty(n, dtype="int64")
        for i in range(n):
            winner = rng.choice(content)
            how_many = length // 2 + 1
            positions = rng.permutation(length)
            x[i, positions[:how_many]] = winner
            labels[i] = winner

    labels = labels - 1          # content token k is class k-1
    class_names = tuple(f"token {k}" for k in content)

    order = rng.permutation(n)
    x, labels = x[order], labels[order]
    cut = int(n * (1.0 - float(spec.val_fraction)))
    y = _one_hot(labels, len(class_names))

    if on_progress:
        on_progress(100, 100, "Done", 0.0)

    return TokenBundle(
        x_train=x[:cut], y_train=y[:cut], x_val=x[cut:], y_val=y[cut:],
        name=TASK_LABELS[task], task=task, length=length, vocab=vocab,
        class_names=class_names,
    )


def majority_baseline(bundle: TokenBundle) -> float:
    """Accuracy of always answering with the commonest class in training."""
    train = np.argmax(bundle.y_train, axis=1)
    winner = int(np.bincount(train, minlength=bundle.n_outputs).argmax())
    return float(np.mean(np.argmax(bundle.y_val, axis=1) == winner))


def baseline(bundle: TokenBundle) -> dict:
    """The give-up score, with the same shape the other workspaces use."""
    return {
        "value": majority_baseline(bundle),
        "metric": "accuracy",
        "name": "majority class",
        "better_is_lower": False,
        "explain": (
            "That is the accuracy of always answering with the commonest token "
            "in the training set - no model, no training. With "
            f"{bundle.n_outputs} possible answers, guessing at random lands near "
            f"{1 / max(bundle.n_outputs, 1):.0%}."
        ),
    }


# ----------------------------------------------------------------- the model

@dataclass
class TransformerConfig:
    """One tiny transformer, sized for CPU training."""

    length: int = 12
    vocab: int = 8
    n_outputs: int = 7
    # Measured on the `match` task over six seeds at 60 epochs: 64 wide with two
    # blocks solved it every time; 32 wide solved it on two of four seeds and got
    # stuck at 0.70 on the others for as long as it was trained; 32 wide split
    # four ways (8 per head) solved it on none. Two blocks, because one block
    # never got past 0.31 - the lookup takes two steps of attention.
    d_model: int = 64
    n_heads: int = 2
    n_blocks: int = 2
    ff_dim: int = 64
    dropout: float = 0.0
    positional: bool = True
    pool: str = MEAN
    learning_rate: float = 0.003
    optimizer: str = "adam"

    @property
    def key_dim(self) -> int:
        """Width of each head. The heads split d_model between them."""
        return max(1, int(self.d_model) // max(1, int(self.n_heads)))

    def stack(self) -> list:
        columns = [{
            "name": "Tokens",
            "detail": f"{self.length} ids, vocabulary {self.vocab}",
            "kind": "input",
        }, {
            "name": "Embed",
            "detail": (f"{self.d_model} wide + positions" if self.positional
                       else f"{self.d_model} wide, NO positions"),
            "kind": "embed",
        }]
        for block in range(1, int(self.n_blocks) + 1):
            columns.append({
                "name": f"Attention {block}",
                "detail": f"{self.n_heads} heads x {self.key_dim}",
                "kind": "attention",
            })
            columns.append({
                "name": f"Feed-forward {block}",
                "detail": f"{self.ff_dim} units",
                "kind": "hidden",
            })
        columns.append({
            "name": "Pool",
            "detail": POOL_LABELS.get(self.pool, self.pool),
            "kind": "hidden",
        })
        columns.append({
            "name": "Answer",
            "detail": f"{self.n_outputs} tokens",
            "kind": "output",
        })
        return columns

    def describe(self) -> str:
        head = (f"{self.n_blocks} block" + ("s" if self.n_blocks > 1 else "")
                + f", {self.n_heads} heads, {self.d_model} wide")
        return head + ("" if self.positional else ", NO positional encoding")


def validate(config: TransformerConfig) -> list:
    problems = []
    if int(config.d_model) % max(1, int(config.n_heads)) != 0:
        problems.append(
            f"{config.d_model} does not divide evenly into {config.n_heads} "
            "heads. The heads share the width between them, so pick a width "
            "that divides"
        )
    if int(config.n_heads) < 1:
        problems.append("At least one attention head is needed")
    if int(config.n_blocks) < 1:
        problems.append("At least one block is needed")
    if config.pool not in POOLS:
        problems.append(f"Unknown pooling '{config.pool}'")
    if int(config.n_outputs) < 2:
        problems.append("A classification head needs at least two classes")
    return problems


def estimate_params(config: TransformerConfig) -> int:
    """Roughly how many weights this will have, before Keras is loaded."""
    width, heads = int(config.d_model), int(config.n_heads)
    key = config.key_dim
    total = int(config.vocab) * width                     # token embedding
    if config.positional:
        total += int(config.length) * width               # position embedding
    for _ in range(int(config.n_blocks)):
        # query, key, value and the output projection, each with its bias
        total += 3 * (width * heads * key + heads * key)
        total += heads * key * width + width
        total += 2 * 2 * width                            # two layer norms
        total += width * int(config.ff_dim) + int(config.ff_dim)
        total += int(config.ff_dim) * width + width
    total += width * int(config.n_outputs) + int(config.n_outputs)
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


def _positions_layer(keras):
    """A learned position embedding added to the tokens.

    Defined inside a function because Keras is imported lazily (invariant 4) and
    a module-level subclass would drag TensorFlow in at import time.
    """

    class AddPositions(keras.layers.Layer):
        def __init__(self, length, width, **kwargs):
            super().__init__(**kwargs)
            self.length = int(length)
            self.embedding = keras.layers.Embedding(int(length), int(width))

        def call(self, inputs):
            return inputs + self.embedding(keras.ops.arange(self.length))

    return AddPositions


def build_transformer(config: TransformerConfig):
    """Compile the model. Returns (model, attention_model).

    The second model shares every weight with the first and outputs the raw
    attention scores instead of an answer, so training one trains both and the
    map on screen is always the map the answer actually came from.
    """
    problems = validate(config)
    if problems:
        raise TransformerError("; ".join(problems))

    keras = keras_module()
    layers = keras.layers
    width = int(config.d_model)

    inputs = keras.Input(shape=(int(config.length),), dtype="int32", name="tokens")
    x = layers.Embedding(int(config.vocab), width, name="token_embedding")(inputs)
    if config.positional:
        x = _positions_layer(keras)(int(config.length), width, name="positions")(x)

    scores = []
    for block in range(int(config.n_blocks)):
        attention = layers.MultiHeadAttention(
            num_heads=int(config.n_heads), key_dim=config.key_dim,
            dropout=float(config.dropout), name=f"attention_{block + 1}",
        )
        attended, score = attention(x, x, return_attention_scores=True)
        scores.append(score)
        x = layers.LayerNormalization(name=f"norm_a_{block + 1}")(x + attended)

        hidden = layers.Dense(int(config.ff_dim), activation="relu",
                              name=f"ff_{block + 1}")(x)
        hidden = layers.Dense(width, name=f"ff_out_{block + 1}")(hidden)
        if config.dropout > 0:
            hidden = layers.Dropout(float(config.dropout),
                                    name=f"drop_{block + 1}")(hidden)
        x = layers.LayerNormalization(name=f"norm_b_{block + 1}")(x + hidden)

    pooled = (layers.GlobalAveragePooling1D(name="pool")(x)
              if config.pool == MEAN
              else layers.Lambda(lambda t: t[:, 0, :], name="pool")(x))
    outputs = layers.Dense(int(config.n_outputs), activation="softmax",
                           name="answer")(pooled)

    model = keras.Model(inputs, outputs, name="tiny_transformer")
    model.compile(
        optimizer=_optimizer(keras, config.optimizer, config.learning_rate),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    attention_model = keras.Model(inputs, scores, name="attention_probe")
    return model, attention_model


def count_params(model) -> tuple:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return trainable, total


# -------------------------------------------------------------- the attention

def attention_maps(attention_model, x: np.ndarray) -> list:
    """Per block, an array of shape (batch, heads, length, length).

    Row i of a map is "when the model was standing at position i, how much did
    it weigh every other position". Each row sums to 1.
    """
    raw = attention_model.predict(x, verbose=0)
    if not isinstance(raw, list):
        raw = [raw]
    return [np.asarray(block, dtype="float32") for block in raw]


def attended_positions(maps: list, block: int = 0) -> np.ndarray:
    """For each sequence, the position the model weighed most overall.

    Averaged over heads and over the querying position, so this is "the column
    that got the most weight", which is what a reader of the heatmap sees.
    """
    weights = np.asarray(maps[block]).mean(axis=1).mean(axis=1)   # (batch, length)
    return np.argmax(weights, axis=1)


def map_sharpness(maps: list, block: int = 0) -> float:
    """How concentrated the map is: 1.0 means one column takes everything.

    Measured as the mean top-column share minus what a flat map would give, then
    rescaled, so a diffuse map lands near 0 and a spike near 1.
    """
    weights = np.asarray(maps[block]).mean(axis=1).mean(axis=1)
    length = weights.shape[1]
    flat = 1.0 / length
    top = weights.max(axis=1).mean()
    return float(max(0.0, (top - flat) / max(1e-9, 1.0 - flat)))


def predict_classes(model, x: np.ndarray) -> np.ndarray:
    return np.argmax(model.predict(x, verbose=0), axis=1)


def accuracy(model, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predict_classes(model, x) == np.argmax(y, axis=1)))


def faithfulness(model, attention_model, bundle: TokenBundle, rng=None,
                 sample: int = 400, block: int = -1) -> dict:
    """Does the attention map point at what the answer actually depends on?

    The test is erasure. Take the position the map weighed most, replace that
    token with a different one, and measure how much accuracy falls. Then do the
    same to some OTHER position, chosen at random, as a control.

    Both edits are the same size and both stay inside the vocabulary, so the
    only difference is which position was touched. If erasing the attended
    position hurts no more than erasing an arbitrary one, the heatmap was not
    describing what the model used - however convincing it looked.

    `block` must be the map the reader is actually looking at. Measured on the
    `match` task with two blocks, block 1's map is diffuse noise while block 2's
    points at the right column 93% of the time; testing the wrong one would
    report a faithful map as decoration, or the reverse. Defaults to the last
    block, the one nearest the answer.
    """
    rng = rng or np.random.default_rng(0)
    take = min(int(sample), len(bundle.x_val))
    x = bundle.x_val[:take].copy()
    y = bundle.y_val[:take]

    maps = attention_maps(attention_model, x)
    block = int(block) % len(maps)
    focus = attended_positions(maps, block)

    rows = np.arange(take)
    content = np.arange(1, bundle.vocab)

    def replaced(positions):
        edited = x.copy()
        # Always a DIFFERENT token, or "erasing" could leave the sequence as it
        # was and quietly understate the damage.
        current = edited[rows, positions]
        swap = rng.choice(content, size=take)
        clash = swap == current
        while clash.any():
            swap[clash] = rng.choice(content, size=int(clash.sum()))
            clash = swap == current
        edited[rows, positions] = swap
        return edited

    other = np.array([
        rng.choice([p for p in range(bundle.length) if p != focus[i]])
        for i in range(take)
    ])

    intact = accuracy(model, x, y)
    hit = accuracy(model, replaced(focus), y)
    control = accuracy(model, replaced(other), y)
    return {
        "intact": intact,
        "attended_erased": hit,
        "control_erased": control,
        "attended_drop": intact - hit,
        "control_drop": intact - control,
        "sharpness": map_sharpness(maps, block),
        "block": block,
        "n": take,
    }
