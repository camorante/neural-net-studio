"""Training runs for the autoencoder workspace, free of any UI framework.

The only structural difference from every other run in this project is the
line that builds the target, and it is worth staring at:

    y = images / 255.0

There is no label. The answer is the question. Everything the network learns
about shapes, edges and colour it learns from being asked to give the picture
back through a hole too small to pass it whole.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .autoencoder import (
    AutoencoderConfig,
    AnomalySplit,
    build_autoencoder,
    count_params,
    format_anomalies,
    mean_image_baseline,
    score_anomalies,
    split_anomaly,
    targets,
)
from .model_builder import keras_module
from .vision import ImageBundle

NO_ANOMALY = -1

# Every arm of a sweep starts from the same weights, so the only thing that
# differs between them is the width of the waist.
SWEEP_SEED = 1234

# Measured on this project's synthetic shapes, 900 images at 32px, conv with
# 2 stages and 16 base filters: all three arms sit on the mean-image baseline
# until about epoch 7, cross 50% apart at epoch 7, and are clearly ordered by
# epoch 15. At 0.7s per epoch, 3 arms x 25 epochs costs under a minute - so the
# preset buys a real result rather than a fast meaningless one.
SWEEP_SIZES = (2, 8, 64)
SWEEP_EPOCHS = 25
SWEEP_BATCH = 32

# Below this the arms have not separated by more than the run-to-run noise we
# measured, so no claim gets made about the bottleneck.
MEANINGFUL_GAIN = 0.15

# Within this much of the mean-image baseline, the model has learned nothing.
LEARNED_NOTHING = 0.10

# The first thing a student notices after a sweep is that the shapes come back
# GREY. The explanation is not a guess - it was measured on this project's
# synthetic shapes (900 images, 32px, conv, 2 stages, 16 base filters), by
# correlating the red-minus-blue balance of each reconstructed shape against
# the original's:
#     latent 64, 25 epochs -> correlation +0.03, 6% of the input's colour
#     latent 64, 90 epochs -> correlation +0.91, 66%
#     latent  8, 90 epochs -> correlation +0.06, 14%
# So at a wide waist colour is only a matter of time, and at a narrow one it
# never arrives at all. Those are two different failures and the note keeps
# them apart rather than blaming "the bottleneck" for both.
COLOUR_NOTE = (
    "  One thing you will notice immediately: the shapes come back but the "
    "COLOUR does not - the reconstructions are grey. That is not a bug, and "
    "which reason applies depends on the waist.\n"
    "  Squared error is dominated by putting a bright shape in the right "
    "place, so that is what gradient descent buys first; colour is a smaller "
    "correction it gets to later. Measured on the synthetic shapes, a latent "
    "of 64 has the room and just needs the time - its colour matches the "
    "original almost perfectly by about 90 epochs, having been essentially "
    "random at 25. A latent of 8 never gets there, not even at 90 epochs: "
    "once the shape is paid for, that waist has nothing left to spend.\n"
    "  So the bottleneck does not blur things evenly. It imposes a priority "
    "order - position, then shape, then colour - and this is you watching it "
    "decide what it can afford."
)


@dataclass
class AutoencoderRequest:
    """One unsupervised training run."""

    data: ImageBundle
    config: AutoencoderConfig
    epochs: int = 20
    batch_size: int = 32
    early_stopping: bool = False
    patience: int = 6
    anomaly_class: int = NO_ANOMALY

    @property
    def holds_out_a_class(self) -> bool:
        return int(self.anomaly_class) != NO_ANOMALY

    def split(self) -> AnomalySplit | None:
        """The anomaly arrangement, or None when every class is trained on."""
        if not self.holds_out_a_class:
            return None
        return split_anomaly(self.data, int(self.anomaly_class))


class AutoencoderRun:
    """Builds an autoencoder, trains it against its own input, reports back."""

    def __init__(self, request: AutoencoderRequest, on_epoch=None, on_message=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def arrays(self) -> tuple:
        """(x_train, x_val, split) after any held-out class is removed."""
        split = self.request.split()
        if split is None:
            return self.request.data.x_train, self.request.data.x_val, None
        return split.x_train, split.x_val, split

    def run(self, model=None, encoder=None, label: str = "") -> dict:
        request = self.request
        config = request.config

        self._say("Loading TensorFlow...")
        keras = keras_module()

        x_train, x_val, split = self.arrays()
        if split is not None:
            self._say(split.describe())

        if model is None:
            kind = "convolutional" if config.variant == "conv" else "dense"
            self._say(
                f"Building a {kind} autoencoder: {config.n_inputs:,} numbers in, "
                f"{config.latent_dim} at the waist ({config.compression:.0f}x)."
            )
            model, encoder = build_autoencoder(config)

        trainable, total = count_params(model)
        self._say(f"{model.name}: {trainable:,} trainable parameters.")
        if config.denoise:
            self._say(
                f"Denoising mode: {config.noise_std:.2f} of the pixel range is "
                "added as noise during training only. The target stays clean, so "
                "the network is being asked to remove damage it has never been "
                "told how to describe."
            )

        run = self

        class _Bridge(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if run.on_epoch:
                    run.on_epoch(
                        epoch + 1,
                        {k: float(v) for k, v in (logs or {}).items()},
                        label,
                    )
                if run._stop:
                    self.model.stop_training = True

            def on_train_batch_end(self, batch, logs=None):
                if run._stop:
                    self.model.stop_training = True

        callbacks = [_Bridge(), keras.callbacks.TerminateOnNaN()]
        validation = None
        if len(x_val) > 0:
            validation = (x_val, targets(x_val))
            if request.early_stopping:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=max(1, request.patience),
                        restore_best_weights=True,
                        verbose=0,
                    )
                )

        self._say(
            f"Training on {len(x_train)} images, validating on {len(x_val)}. "
            "No labels are used anywhere in this run."
        )
        history = model.fit(
            x_train,
            targets(x_train),
            validation_data=validation,
            epochs=max(1, request.epochs),
            batch_size=max(1, request.batch_size),
            verbose=0,
            callbacks=callbacks,
        )

        result = {
            "model": model,
            "encoder": encoder,
            "label": label or f"latent {config.latent_dim}",
            "history": {
                k: [float(v) for v in vals] for k, vals in history.history.items()
            },
            "epochs_run": len(history.history.get("loss", [])),
            "trainable_params": trainable,
            "total_params": total,
            "latent_dim": int(config.latent_dim),
            "compression": float(config.compression),
            "variant": config.variant,
            "denoise": bool(config.denoise),
            "stopped": self._stop,
            # The score for giving up, so every loss on screen has a floor to
            # be read against instead of being a bare number.
            "baseline": mean_image_baseline(x_train, x_val),
        }
        if validation is not None:
            scores = model.evaluate(
                x_val, targets(x_val), verbose=0, return_dict=True
            )
            result["final_scores"] = {k: float(v) for k, v in scores.items()}

        if split is not None and not self._stop:
            self._say("Scoring the held-out class against the familiar ones...")
            result["anomaly"] = score_anomalies(model, split)
        return result


class LatentSweepRun:
    """Trains the same autoencoder at several latent sizes, back to back.

    Same images, same depth, same initial weights, same number of epochs. The
    only thing that changes is how many numbers the picture has to fit through,
    which makes the bottleneck the single variable under test.
    """

    def __init__(self, request: AutoencoderRequest, latent_sizes, on_epoch=None,
                 on_message=None, on_arm_done=None):
        sizes = sorted({max(1, int(v)) for v in latent_sizes})
        if len(sizes) < 2:
            raise ValueError("A sweep needs at least two different latent sizes")
        self.request = request
        self.latent_sizes = sizes
        self.on_epoch = on_epoch
        self.on_message = on_message
        self.on_arm_done = on_arm_done
        self._stop = False
        self._current: AutoencoderRun | None = None

    def stop(self) -> None:
        self._stop = True
        if self._current:
            self._current.stop()

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def run(self) -> dict:
        keras = keras_module()
        arms = []

        for latent in self.latent_sizes:
            if self._stop:
                break
            config = replace(self.request.config, latent_dim=latent)
            self._say(
                f"\n--- LATENT {latent} "
                f"({config.compression:.0f}x compression) ---"
            )
            # Identical initialisation across arms: the waist is the variable.
            keras.utils.set_random_seed(SWEEP_SEED)
            sub = AutoencoderRun(
                replace(self.request, config=config),
                on_epoch=self.on_epoch,
                on_message=self.on_message,
            )
            self._current = sub
            outcome = sub.run(label=f"latent {latent}")
            self._current = None
            arms.append(outcome)
            if self.on_arm_done:
                self.on_arm_done(outcome["label"], outcome)

        return {
            "arms": arms,
            "stopped": self._stop,
            "variant": self.request.config.variant,
            "n_inputs": self.request.config.n_inputs,
        }


def _final_val_loss(arm: dict) -> float:
    scores = arm.get("final_scores") or {}
    if "loss" in scores:
        return float(scores["loss"])
    losses = arm.get("history", {}).get("val_loss") or arm.get("history", {}).get("loss")
    return float(losses[-1]) if losses else float("nan")


def _baseline_of(arms: list) -> float:
    """The mean-image floor. Every arm shares one dataset, so one value."""
    for arm in arms:
        value = arm.get("baseline")
        if value is not None and value == value:  # not NaN
            return float(value)
    return float("nan")


def format_run(result: dict) -> str:
    """A single run's readout, with the score for giving up alongside it."""
    loss = _final_val_loss(result)
    baseline = result.get("baseline", float("nan"))
    lines = [
        f"Latent {result['latent_dim']} "
        f"({result['compression']:.0f}x squeeze), "
        f"{result['epochs_run']} epochs, "
        f"{result['trainable_params']:,} trainable parameters.",
        f"  val MSE {loss:.5f}",
    ]
    if baseline == baseline:
        lines.append(
            f"  predicting the average image scores {baseline:.5f} - that is "
            "the floor that means nothing was learned."
        )
        if loss > baseline * (1 - LEARNED_NOTHING):
            lines.append("")
            lines.append(
                "  THIS RUN HAS LEARNED ESSENTIALLY NOTHING YET."
                if loss <= baseline
                else "  THIS RUN IS STILL WORSE THAN GIVING UP."
            )
            lines.append(
                "  A fresh decoder starts by answering mid-grey, and these "
                "images are mostly dark, so the first thing it has to learn is "
                "the overall brightness - not the shapes. Until the error drops "
                "below the give-up floor, nothing on screen is a reconstruction."
                if loss > baseline
                else "  The error is sitting on the give-up floor, which is what "
                "an autoencoder outputs when it answers every image with the "
                "same blur."
            )
            lines.append(
                f"  Give it more epochs - around {SWEEP_EPOCHS} is where the "
                "shapes start appearing on this dataset."
            )
        else:
            factor = baseline / max(loss, 1e-12)
            lines.append(
                f"  So the reconstruction is {factor:.1f}x better than giving up."
            )
    if result.get("denoise"):
        lines.append("")
        lines.append(
            "  Denoising run: the input was corrupted and the target was not, "
            "so this error is measured against the CLEAN image. The network "
            "learned to remove damage nobody described to it."
        )
    return "\n".join(lines)


def format_sweep(result: dict) -> str:
    """A readout of the latent sweep that states only what was measured."""
    arms = result.get("arms") or []
    if len(arms) < 2:
        return "Sweep incomplete - at least two arms are needed to compare."

    n_inputs = int(result.get("n_inputs") or 0)
    baseline = _baseline_of(arms)
    lines = [
        f"Same images, same depth, same initial weights. "
        f"{n_inputs:,} numbers go in; only the waist changed.\n"
    ]
    for arm in arms:
        lines.append(
            f"  latent {arm['latent_dim']:>4}  "
            f"({arm['compression']:>6.0f}x squeeze):  "
            f"val MSE {_final_val_loss(arm):.5f}  "
            f"({arm['trainable_params']:,} params, {arm['epochs_run']} epochs)"
        )
    if baseline == baseline:
        lines.append(
            f"  {'giving up':>12}  {'':>14}   val MSE {baseline:.5f}  "
            "(answer every image with the average)"
        )

    ordered = sorted(arms, key=lambda a: a["latent_dim"])
    tightest, widest = ordered[0], ordered[-1]
    tight_loss, wide_loss = _final_val_loss(tightest), _final_val_loss(widest)
    losses = [_final_val_loss(a) for a in ordered]
    lines.append("")

    # Nothing may be claimed about the bottleneck while every arm is still
    # sitting on the floor. This guard exists because a 2-epoch sweep looks
    # like a tie and would otherwise be reported as one.
    if baseline == baseline and min(losses) > baseline * (1 - LEARNED_NOTHING):
        lines.append(
            "  NO CONCLUSION IS AVAILABLE FROM THIS RUN. Not one arm has beaten "
            "the give-up floor, so they are not tied - none of them has started "
            "learning. This says nothing whatsoever about the bottleneck."
        )
        if min(losses) > baseline:
            lines.append(
                "  In fact every arm is still WORSE than answering with the "
                "average image, because a fresh decoder starts at mid-grey and "
                "has to learn the overall brightness before any shape appears."
            )
        lines.append(
            f"  Run the sweep again with about {SWEEP_EPOCHS} epochs. On this "
            "dataset the arms are still identical at epoch 2 and only pull "
            "apart around epoch 7."
        )
        return "\n".join(lines)

    gain = (tight_loss - wide_loss) / max(tight_loss, 1e-12)
    if gain >= MEANINGFUL_GAIN:
        factor = tight_loss / max(wide_loss, 1e-12)
        lines.append(
            f"  Widening the waist from {tightest['latent_dim']} to "
            f"{widest['latent_dim']} cut the error {factor:.1f}x. The bottleneck "
            "is a budget, and the reconstruction rows show what it bought."
        )
    elif gain <= -MEANINGFUL_GAIN:
        lines.append(
            f"  The WIDER waist ({widest['latent_dim']}) did worse than the "
            f"narrow one ({tightest['latent_dim']}). More capacity is not "
            "automatically more skill - a wider latent layer also means more "
            "parameters to fit in the same number of epochs, and here that cost "
            "more than the extra room was worth."
        )
    else:
        lines.append(
            f"  The arms did NOT separate: {tightest['latent_dim']} and "
            f"{widest['latent_dim']} finished within "
            f"{abs(gain) * 100:.0f}% of each other, which is inside the "
            "run-to-run noise. Either train longer, or accept that on this "
            "dataset the narrow waist was already enough - both are real "
            "answers, and neither is the one the textbook picture predicts."
        )

    monotonic = all(later <= earlier for earlier, later in zip(losses, losses[1:]))
    if not monotonic:
        lines.append(
            "  The error did not fall cleanly with every step, so the ordering "
            "of the middle arms is not trustworthy from this single run."
        )

    lines.append("")
    lines.append(
        "  Look at the reconstruction rows, not only these numbers. A tight "
        "waist does not fail randomly: it keeps position and rough size and "
        "throws away edges and detail. What survives the squeeze is what the "
        "network judged the image to be."
    )
    lines.append("")
    lines.append(COLOUR_NOTE)
    return "\n".join(lines)


def hardest_to_reconstruct(errors: np.ndarray, count: int) -> np.ndarray:
    """Indices of the images the model rebuilt worst, worst first."""
    errors = np.asarray(errors, dtype="float32")
    if len(errors) == 0:
        return np.zeros(0, dtype="int64")
    count = int(min(max(1, count), len(errors)))
    return np.argsort(-errors)[:count].astype("int64")


__all__ = [
    "COLOUR_NOTE",
    "LEARNED_NOTHING",
    "MEANINGFUL_GAIN",
    "NO_ANOMALY",
    "SWEEP_BATCH",
    "SWEEP_EPOCHS",
    "SWEEP_SIZES",
    "AutoencoderRequest",
    "AutoencoderRun",
    "LatentSweepRun",
    "format_anomalies",
    "format_run",
    "format_sweep",
    "hardest_to_reconstruct",
]
