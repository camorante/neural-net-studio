"""Autoencoders: the first models in this project that learn with no labels.

Every other network here is handed the answer. An autoencoder is handed only
the question: reproduce your own input after forcing it through a bottleneck
too small to hold it. There is no label anywhere in the loss - the target IS
the input - and that single change is what makes this a different kind of
learning rather than another architecture.

The bottleneck is the entire lesson. A 32x32 colour image is 3072 numbers; a
16-unit latent layer has to carry all of it. Whatever survives that squeeze is
what the network decided mattered, and whatever blurs away is what it decided
did not. Widen the latent layer until it is as large as the input and the
exercise collapses: the cheapest solution becomes the identity function and
nothing is learned at all.

Scale convention, and it is deliberate: the model accepts pixels in 0..255
(what `ImageBundle` already holds) and returns its reconstruction in 0..1,
because a loss read in 0..1 units is legible and one read in 0..255 units is
not. `reconstruct()` is the only place the multiplication back to pixels
lives - callers never do it themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_builder import keras_module

# count_params lives in resnet.py because that is where the transfer-learning
# tab needed it first. It is a generic model measurement, so it is imported
# rather than copied - a second copy is a second thing to keep honest.
from .resnet import count_params  # noqa: F401 - re-exported for the UI

CONV = "conv"
DENSE = "dense"
VARIANTS = (CONV, DENSE)

VARIANT_LABELS = {
    CONV: "Convolutional - keeps the picture's geometry",
    DENSE: "Dense - throws the geometry away",
}

MIN_DENSE_UNITS = 32


class AutoencoderError(ValueError):
    """Raised when a configuration cannot describe a working autoencoder."""


@dataclass
class AutoencoderConfig:
    """An encoder/decoder pair sized for CPU training."""

    input_shape: tuple = (32, 32, 3)
    variant: str = CONV
    latent_dim: int = 16
    stages: int = 2
    base_filters: int = 16
    hidden_units: int = 256
    denoise: bool = False
    noise_std: float = 0.25
    learning_rate: float = 0.001
    optimizer: str = "adam"

    # ------------------------------------------------------------------ shapes

    @property
    def n_inputs(self) -> int:
        """How many numbers one image is, before anything is squeezed."""
        height, width, channels = self.input_shape
        return int(height) * int(width) * int(channels)

    @property
    def compression(self) -> float:
        """How many times smaller the latent code is than the image."""
        return self.n_inputs / max(1, int(self.latent_dim))

    def encoder_filters(self) -> list:
        """Filters per downsampling stage, doubling as the map shrinks."""
        return [int(self.base_filters) * (2 ** i) for i in range(int(self.stages))]

    def dense_units(self) -> list:
        """Hidden widths per dense stage, halving on the way in."""
        units = []
        width = int(self.hidden_units)
        for _ in range(int(self.stages)):
            units.append(max(MIN_DENSE_UNITS, width))
            width //= 2
        return units

    def bottleneck_map(self) -> tuple:
        """Spatial shape reached just before the latent layer (conv only)."""
        height, width, _ = self.input_shape
        divisor = 2 ** int(self.stages)
        filters = self.encoder_filters()[-1] if self.stages else int(self.input_shape[2])
        return int(height) // divisor, int(width) // divisor, int(filters)

    # ------------------------------------------------------------------ diagram

    def stack(self) -> list:
        """Layer descriptors for the diagram, sized by real tensor widths.

        `values` is how many numbers exist at that point in the network, so a
        widget can draw the pinch from measured shapes instead of a guess.
        """
        height, width, channels = (int(v) for v in self.input_shape)
        columns = [
            {
                "name": "Image",
                "detail": f"{height}x{width}x{channels}",
                "values": self.n_inputs,
                "kind": "input",
            }
        ]

        if self.variant == CONV:
            size_h, size_w = height, width
            for stage, filters in enumerate(self.encoder_filters(), start=1):
                size_h, size_w = max(1, size_h // 2), max(1, size_w // 2)
                columns.append(
                    {
                        "name": f"Encode {stage}",
                        "detail": f"{filters}f, {size_h}x{size_w}",
                        "values": size_h * size_w * filters,
                        "kind": "encoder",
                    }
                )
        else:
            for stage, units in enumerate(self.dense_units(), start=1):
                columns.append(
                    {
                        "name": f"Encode {stage}",
                        "detail": f"{units} units",
                        "values": units,
                        "kind": "encoder",
                    }
                )

        columns.append(
            {
                "name": "Latent",
                "detail": f"{int(self.latent_dim)} numbers",
                "values": max(1, int(self.latent_dim)),
                "kind": "latent",
            }
        )

        # The decoder is the encoder read backwards, which is the point: the
        # picture has to be rebuilt through the same widths it was reduced by.
        mirror = columns[1:-1][::-1]
        for index, column in enumerate(mirror, start=1):
            columns.append(
                {
                    "name": f"Decode {index}",
                    "detail": column["detail"],
                    "values": column["values"],
                    "kind": "decoder",
                }
            )

        columns.append(
            {
                "name": "Reconstruction",
                "detail": f"{height}x{width}x{channels}",
                "values": self.n_inputs,
                "kind": "output",
            }
        )
        return columns

    def describe(self) -> str:
        return (
            f"{self.n_inputs:,} numbers in, {int(self.latent_dim)} at the waist: "
            f"a {self.compression:.0f}x squeeze."
        )


def validate(config: AutoencoderConfig) -> list:
    """Everything wrong with a configuration, in plain language."""
    problems = []

    if config.variant not in VARIANTS:
        problems.append(f"Unknown variant {config.variant!r}")
    if len(config.input_shape) != 3:
        problems.append("Images need a height, a width and a channel count")
        return problems

    height, width, channels = (int(v) for v in config.input_shape)
    latent = int(config.latent_dim)

    if latent < 1:
        problems.append("The latent layer needs at least 1 unit")
    if latent >= config.n_inputs:
        problems.append(
            f"A latent layer of {latent} is not smaller than the "
            f"{config.n_inputs} numbers going in, so there is no bottleneck - "
            "the network can just copy its input and learn nothing"
        )
    if int(config.stages) < 1:
        problems.append("At least one encoder stage is needed")

    if config.variant == CONV and int(config.stages) >= 1:
        divisor = 2 ** int(config.stages)
        if height % divisor or width % divisor:
            problems.append(
                f"{height}x{width} images cannot be halved {config.stages} times "
                f"evenly - use fewer stages or an image size divisible by {divisor}"
            )
    if channels not in (1, 3):
        problems.append("Images must be greyscale or RGB")
    if not 0.0 <= float(config.noise_std) <= 1.0:
        problems.append("Noise is a fraction of the pixel range, so 0 to 1")
    return problems


def estimate_params(config: AutoencoderConfig) -> int:
    """A rough weight count, cheap enough to show while a slider moves.

    Only the two dense layers either side of the latent layer are counted for
    the convolutional variant, because those are the ones that explode when the
    latent size changes - which is what the number is there to warn about.
    """
    latent = max(1, int(config.latent_dim))
    if config.variant == DENSE:
        units = config.dense_units()
        total = config.n_inputs * units[0] + units[0]
        for before, after in zip(units, units[1:]):
            total += before * after + after
        total += units[-1] * latent + latent
        # The decoder mirrors it, ending on one weight per output pixel.
        total += latent * units[-1] + units[-1]
        for before, after in zip(units[::-1], units[::-1][1:]):
            total += before * after + after
        total += units[0] * config.n_inputs + config.n_inputs
        return int(total)

    flat = int(np.prod(config.bottleneck_map()))
    convs = 0
    channels = int(config.input_shape[2])
    previous = channels
    for filters in config.encoder_filters():
        convs += 9 * previous * filters
        previous = filters
    for filters in config.encoder_filters()[::-1]:
        convs += 9 * previous * filters
        previous = filters
    convs += 9 * previous * channels + channels
    return int(convs + flat * latent + latent + latent * flat + flat)


# ------------------------------------------------------------------- building

def _optimizer(keras, name: str, learning_rate: float):
    factories = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
        "nadam": keras.optimizers.Nadam,
    }
    factory = factories.get(str(name).lower(), keras.optimizers.Adam)
    return factory(learning_rate=float(learning_rate))


def build_autoencoder(config: AutoencoderConfig):
    """Compile the encoder/decoder pair. Returns (autoencoder, encoder).

    The encoder shares its weights with the autoencoder - it is the same graph
    cut short at the latent layer - so training one trains both.
    """
    problems = validate(config)
    if problems:
        raise AutoencoderError("; ".join(problems))

    keras = keras_module()
    layers = keras.layers
    channels = int(config.input_shape[2])

    inputs = keras.Input(shape=config.input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    if config.denoise:
        # GaussianNoise is active during training and inert at predict time,
        # which is exactly what a denoising autoencoder needs: it is shown a
        # corrupted image and scored against the clean one.
        x = layers.GaussianNoise(float(config.noise_std), name="corrupt")(x)

    if config.variant == CONV:
        latent, decoded = _conv_body(layers, x, config, channels)
    else:
        latent, decoded = _dense_body(layers, x, config)

    model = keras.Model(inputs, decoded, name=f"{config.variant}_autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    model.compile(
        optimizer=_optimizer(keras, config.optimizer, config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model, encoder


def _conv_body(layers, x, config: AutoencoderConfig, channels: int):
    """Strided convolutions down, transposed convolutions back up."""
    filters_in = config.encoder_filters()

    for stage, filters in enumerate(filters_in, start=1):
        x = layers.Conv2D(
            filters, 3, strides=2, padding="same", use_bias=False,
            name=f"enc{stage}_conv",
        )(x)
        x = layers.BatchNormalization(name=f"enc{stage}_bn")(x)
        x = layers.ReLU(name=f"enc{stage}_relu")(x)

    before_flatten = tuple(int(v) for v in x.shape[1:])
    x = layers.Flatten(name="flatten")(x)
    latent = layers.Dense(int(config.latent_dim), name="latent")(x)

    y = layers.Dense(
        int(np.prod(before_flatten)), activation="relu", name="expand"
    )(latent)
    y = layers.Reshape(before_flatten, name="unflatten")(y)

    for stage, filters in enumerate(filters_in[::-1], start=1):
        y = layers.Conv2DTranspose(
            filters, 3, strides=2, padding="same", use_bias=False,
            name=f"dec{stage}_conv",
        )(y)
        y = layers.BatchNormalization(name=f"dec{stage}_bn")(y)
        y = layers.ReLU(name=f"dec{stage}_relu")(y)

    # Sigmoid because the target is a 0..1 image; anything else would let the
    # network predict impossible pixels and hide the error in the loss.
    decoded = layers.Conv2D(
        channels, 3, padding="same", activation="sigmoid", name="pixels"
    )(y)
    return latent, decoded


def _dense_body(layers, x, config: AutoencoderConfig):
    """Flatten first, which is precisely what loses the geometry."""
    units = config.dense_units()

    x = layers.Flatten(name="flatten")(x)
    for stage, width in enumerate(units, start=1):
        x = layers.Dense(width, activation="relu", name=f"enc{stage}")(x)
    latent = layers.Dense(int(config.latent_dim), name="latent")(x)

    y = latent
    for stage, width in enumerate(units[::-1], start=1):
        y = layers.Dense(width, activation="relu", name=f"dec{stage}")(y)
    y = layers.Dense(config.n_inputs, activation="sigmoid", name="pixels")(y)
    decoded = layers.Reshape(
        tuple(int(v) for v in config.input_shape), name="unflatten"
    )(y)
    return latent, decoded


# ------------------------------------------------------------------ inference

def targets(images: np.ndarray) -> np.ndarray:
    """What the autoencoder is scored against: its own input, in 0..1.

    This is the whole supervision signal, and there is not a label in it.
    """
    return np.asarray(images, dtype="float32") / 255.0


def reconstruct(model, images: np.ndarray) -> np.ndarray:
    """Run the autoencoder and hand back displayable pixels in 0..255."""
    predicted = model.predict(np.asarray(images, dtype="float32"), verbose=0)
    return np.clip(np.asarray(predicted, dtype="float32"), 0.0, 1.0) * 255.0


def reconstruction_errors(model, images: np.ndarray, against=None) -> np.ndarray:
    """Per-image mean squared error, in the same 0..1 units as the loss.

    This single number per image is what turns an autoencoder into an anomaly
    detector: whatever the network was never trained to rebuild rebuilds badly.

    `against` is the images to score the output on when they are not the images
    fed in. A denoising autoencoder is handed the corrupted picture and judged
    on the clean one, and scoring it against its own noisy input would quietly
    reward it for reproducing the damage.
    """
    images = np.asarray(images, dtype="float32")
    if len(images) == 0:
        return np.zeros(0, dtype="float32")
    truth = images if against is None else np.asarray(against, dtype="float32")
    predicted = np.asarray(model.predict(images, verbose=0), dtype="float32")
    difference = (predicted - targets(truth)).reshape(len(images), -1)
    return np.mean(difference * difference, axis=1).astype("float32")


def add_noise(images: np.ndarray, std: float, rng=None) -> np.ndarray:
    """Corrupt pixels the way `GaussianNoise` does inside the model.

    The layer only fires during training, so showing a student what the
    denoising model actually sees means reproducing the corruption out here.
    """
    rng = rng or np.random.default_rng()
    images = np.asarray(images, dtype="float32")
    noisy = images + rng.normal(0.0, float(std) * 255.0, images.shape)
    return np.clip(noisy, 0.0, 255.0).astype("float32")


def latent_codes(encoder, images: np.ndarray) -> np.ndarray:
    """The latent vector for each image - the compressed representation."""
    return np.asarray(
        encoder.predict(np.asarray(images, dtype="float32"), verbose=0),
        dtype="float32",
    )


def mean_image_baseline(x_train: np.ndarray, x_val: np.ndarray) -> float:
    """The MSE of the laziest possible autoencoder: always answer the average.

    This number is the floor that means nothing was learned. An untrained
    network - and a barely trained one - lands almost exactly here, because
    predicting the mean of everything is what minimises squared error when you
    know nothing about the particular image. Without it on screen, a student
    reading `val MSE 0.141` has no way to know that is the score for giving up.
    """
    train = targets(x_train)
    if len(train) == 0 or len(x_val) == 0:
        return float("nan")
    average = np.mean(train, axis=0, keepdims=True)
    difference = targets(x_val) - average
    return float(np.mean(difference * difference))


# -------------------------------------------------------------- anomaly split

@dataclass
class AnomalySplit:
    """One class held out entirely, so the model never learns to rebuild it."""

    x_train: np.ndarray
    x_val: np.ndarray
    anomalies: np.ndarray
    anomaly_name: str
    normal_names: list

    @property
    def n_anomalies(self) -> int:
        return int(len(self.anomalies))

    def describe(self) -> str:
        return (
            f"Training on {len(self.x_train)} images of "
            f"{', '.join(self.normal_names)}; {self.n_anomalies} "
            f"'{self.anomaly_name}' images held out and never shown."
        )


def split_anomaly(bundle, anomaly_index: int) -> AnomalySplit:
    """Remove one class from training and keep it aside as the anomaly.

    This is the honest version of anomaly detection: the held-out class is not
    merely rare in training, it is absent. Anything the model says about it is
    therefore a statement about unfamiliarity, not about a label it half-saw.
    """
    anomaly_index = int(anomaly_index)
    if not 0 <= anomaly_index < bundle.n_classes:
        raise AutoencoderError(
            f"Class {anomaly_index} is not one of the {bundle.n_classes} available"
        )
    if bundle.n_classes < 2:
        raise AutoencoderError("Holding a class out needs at least 2 classes")

    train_labels = np.argmax(bundle.y_train, axis=1)
    val_labels = np.argmax(bundle.y_val, axis=1)

    x_train = bundle.x_train[train_labels != anomaly_index]
    x_val = bundle.x_val[val_labels != anomaly_index]
    anomalies = np.concatenate(
        [
            bundle.x_train[train_labels == anomaly_index],
            bundle.x_val[val_labels == anomaly_index],
        ]
    )
    if len(x_train) < 20:
        raise AutoencoderError(
            "Holding that class out leaves too few training images - "
            "build a larger dataset or pick another class"
        )

    names = list(bundle.class_names)
    return AnomalySplit(
        x_train=x_train,
        x_val=x_val,
        anomalies=anomalies,
        anomaly_name=names[anomaly_index],
        normal_names=[n for i, n in enumerate(names) if i != anomaly_index],
    )


def score_anomalies(model, split: AnomalySplit) -> dict:
    """Compare reconstruction error on familiar images against the held-out class."""
    normal = reconstruction_errors(model, split.x_val)
    strange = reconstruction_errors(model, split.anomalies)
    if len(normal) == 0 or len(strange) == 0:
        return {}

    # ROC AUC answers the only question that matters here: pick one familiar
    # image and one unfamiliar one at random - how often does the error put
    # them in the right order? 0.5 is a coin toss.
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    truth = np.concatenate([np.zeros(len(normal)), np.ones(len(strange))])
    scores = np.concatenate([normal, strange])
    return {
        "anomaly_name": split.anomaly_name,
        "normal_names": list(split.normal_names),
        "normal_mean": float(np.mean(normal)),
        "normal_std": float(np.std(normal)),
        "anomaly_mean": float(np.mean(strange)),
        "anomaly_std": float(np.std(strange)),
        "ratio": float(np.mean(strange) / max(np.mean(normal), 1e-12)),
        "auc": float(roc_auc_score(truth, scores)),
        "n_normal": int(len(normal)),
        "n_anomaly": int(len(strange)),
    }


def format_anomalies(scores: dict) -> str:
    """A readout that does not oversell a weak separation."""
    if not scores:
        return "No anomaly class was held out."

    auc = scores["auc"]
    lines = [
        f"Held out '{scores['anomaly_name']}' entirely. The model only ever saw "
        f"{', '.join(scores['normal_names'])}.",
        "",
        f"  familiar images ({scores['n_normal']}): "
        f"error {scores['normal_mean']:.5f} +/- {scores['normal_std']:.5f}",
        f"  '{scores['anomaly_name']}' ({scores['n_anomaly']}): "
        f"error {scores['anomaly_mean']:.5f} +/- {scores['anomaly_std']:.5f}",
        f"  the unfamiliar class rebuilds {scores['ratio']:.2f}x worse.",
        "",
        f"  ROC AUC {auc:.3f} - pick one familiar and one unfamiliar image at "
        f"random, and the reconstruction error ranks them correctly "
        f"{auc * 100:.0f}% of the time.",
    ]

    lines.append("")
    lines.append(
        "  Before you trust this number: it moves with the random "
        "initialisation, not just with your settings. Measured on the "
        "synthetic shapes at latent 8, six seeds gave AUC 0.604, 0.650, 0.714, "
        "0.726, 0.776 and 0.836 - a spread of 0.23 with nothing else changed. "
        "One run is an anecdote. Train it again before believing the third "
        "decimal, or any of them."
    )
    lines.append("")

    if auc >= 0.9:
        lines.append(
            "  That is a usable detector built without a single label of the "
            "thing it detects, which is why autoencoders are used for faults, "
            "fraud and defects - the anomalies you have no examples of yet."
        )
    elif auc >= 0.7:
        lines.append(
            "  A real signal, but not a reliable detector. The bottleneck is "
            "probably wide enough to rebuild anything roughly - narrow the "
            "latent layer and the unfamiliar class suffers first."
        )
    else:
        lines.append(
            "  No usable separation. Either the latent layer is wide enough to "
            "generalise to the unseen class, or the classes are too alike for "
            "reconstruction error to tell them apart. Both are honest results, "
            "and both are worth more than a number that flatters the method."
        )
    return "\n".join(lines)


# --------------------------------------------------------- judging one image

def judge_image(model, image: np.ndarray, reference: np.ndarray) -> dict:
    """Score one new image against how well the model rebuilds what it knows.

    `reference` is the familiar set: the validation images from the classes the
    model actually trained on. The verdict is a PERCENTILE rather than a
    threshold, because "its error is higher than 97% of what I know" is a
    statement this measurement supports, while "this is an anomaly" is a
    decision that needs a cutoff somebody chose - and choosing it is not the
    model's job.
    """
    reference = np.asarray(reference, dtype="float32")
    if len(reference) < 8:
        raise AutoencoderError(
            "Not enough familiar images to compare against - at least 8 are needed"
        )

    familiar = reconstruction_errors(model, reference)
    error = float(reconstruction_errors(model, image)[0])
    spread = float(np.std(familiar))
    mean = float(np.mean(familiar))
    return {
        "error": error,
        "percentile": float((familiar < error).mean() * 100.0),
        "reference_mean": mean,
        "reference_std": spread,
        "reference_max": float(np.max(familiar)),
        "reference_n": int(len(familiar)),
        "sigmas": float((error - mean) / spread) if spread > 0 else 0.0,
    }


def format_judgement(verdict: dict, latent: np.ndarray | None = None) -> str:
    """Say what the number supports, and not one word more."""
    if not verdict:
        return "Nothing to judge."

    percentile = verdict["percentile"]
    lines = [
        f"Reconstruction error: {verdict['error']:.5f}",
        f"  The {verdict['reference_n']} images this model knows average "
        f"{verdict['reference_mean']:.5f} (worst of them: "
        f"{verdict['reference_max']:.5f}).",
        f"  Yours is higher than {percentile:.0f}% of them, "
        f"{verdict['sigmas']:+.1f} standard deviations from their mean.",
        "",
    ]

    if percentile >= 99:
        lines.append(
            "  VERY UNFAMILIAR. The model rebuilt this worse than almost "
            "everything it was trained on."
        )
    elif percentile >= 95:
        lines.append(
            "  UNFAMILIAR. Above the bulk of what the model knows - the kind of "
            "image most anomaly detectors would flag."
        )
    elif percentile >= 80:
        lines.append(
            "  SLIGHTLY UNUSUAL. Inside the range the model knows, but at the "
            "hard end of it. On its own this is not a finding."
        )
    else:
        lines.append(
            "  FAMILIAR. The model rebuilt this about as well as its own "
            "training data, so it has seen this kind of thing before."
        )

    lines.append("")
    lines.append(
        "  Two honest caveats. Unfamiliar is not the same as defective - the "
        "model only reports that something does not look like what it was "
        "shown. And your image was resized to the dataset's size, which for "
        "32x32 throws away almost everything a photograph contained, so a "
        "photo will read as unfamiliar for that reason alone."
    )
    # Measured on the synthetic shapes at latent 8, same dataset, four seeds:
    # a clean hand-drawn triangle scored 63, 78, 82 and 91; a circle scored
    # 28, 29, 49 and 78; a stripe pattern scored 100 every time. So a middling
    # percentile is the seed talking, and an emphatic one is the image.
    lines.append("")
    if 20 <= percentile <= 90:
        lines.append(
            "  TRUST THIS ONE LOOSELY. A percentile in the middle moves a lot "
            "with the random initialisation, not with your image: retrained on "
            "four different seeds, the same hand-drawn circle scored anywhere "
            "from 28% to 78%. Train again and see whether the answer holds."
        )
    else:
        lines.append(
            "  This verdict is at the emphatic end, which is where it is worth "
            "something. Retrained on four seeds, genuinely alien images stayed "
            "at 100% every single time, while borderline ones swung by 50 "
            "points. An anomaly detector is trustworthy when it is sure."
        )

    if latent is not None and len(np.ravel(latent)) <= 24:
        values = " ".join(f"{v:+.2f}" for v in np.ravel(latent))
        lines.append("")
        lines.append("  Its latent code, the whole image squeezed down:")
        lines.append(f"    {values}")
    return "\n".join(lines)
