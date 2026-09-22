"""Training runs for the convolutional tab, free of any UI framework."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .model_builder import keras_module
from .resnet import (
    ResNetConfig,
    TransferConfig,
    build_resnet,
    build_transfer,
    count_params,
)
from .vision import ImageBundle

SCRATCH = "scratch"
TRANSFER = "transfer"


@dataclass
class VisionRequest:
    """One convolutional training run."""

    data: ImageBundle
    config: object  # ResNetConfig or TransferConfig
    epochs: int = 20
    batch_size: int = 32
    early_stopping: bool = False
    patience: int = 6

    @property
    def mode(self) -> str:
        return TRANSFER if isinstance(self.config, TransferConfig) else SCRATCH


class VisionRun:
    """Builds a convolutional model, trains it, reports through callbacks."""

    def __init__(self, request: VisionRequest, on_epoch=None, on_message=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def build(self):
        request = self.request
        if request.mode == TRANSFER:
            self._say(
                f"Loading {request.config.backbone} with ImageNet weights "
                "(first run downloads ~98 MB)..."
            )
            return build_transfer(request.config)
        kind = "residual" if request.config.use_skip else "plain (no skip connections)"
        self._say(f"Building a {kind} stack, {request.config.depth} weight layers deep...")
        return build_resnet(request.config)

    def run(self, model=None, label: str = "") -> dict:
        request = self.request
        data = request.data

        self._say("Loading TensorFlow...")
        keras = keras_module()

        model = model or self.build()
        trainable, total = count_params(model)
        self._say(
            f"{model.name}: {trainable:,} trainable of {total:,} total parameters."
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
        if data.n_val > 0:
            validation = (data.x_val, data.y_val)
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
            f"Training on {data.n_train} images, validating on {data.n_val}."
        )
        history = model.fit(
            data.x_train,
            data.y_train,
            validation_data=validation,
            epochs=max(1, request.epochs),
            batch_size=max(1, request.batch_size),
            verbose=0,
            callbacks=callbacks,
        )

        result = {
            "model": model,
            "label": label or model.name,
            "history": {
                k: [float(v) for v in vals] for k, vals in history.history.items()
            },
            "epochs_run": len(history.history.get("loss", [])),
            "trainable_params": trainable,
            "total_params": total,
            "stopped": self._stop,
            "mode": request.mode,
        }
        if data.n_val > 0:
            scores = model.evaluate(
                data.x_val, data.y_val, verbose=0, return_dict=True
            )
            result["final_scores"] = {k: float(v) for k, v in scores.items()}
        return result


class SkipComparisonRun:
    """Trains the same stack twice: with skip connections, then without.

    Same depth, same width, same data, same seed. The only difference is the
    addition inside each block, which isolates what the shortcut actually buys.
    """

    def __init__(self, request: VisionRequest, on_epoch=None, on_message=None,
                 on_arm_done=None):
        if not isinstance(request.config, ResNetConfig):
            raise ValueError("The skip comparison only applies to a from-scratch stack")
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self.on_arm_done = on_arm_done
        self._stop = False
        self._current: VisionRun | None = None

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
        for label, use_skip in (("residual", True), ("plain", False)):
            if self._stop:
                break
            self._say(
                f"\n--- {label.upper()} arm "
                f"({'with' if use_skip else 'without'} skip connections) ---"
            )
            # Identical initialisation for both arms, so the comparison is fair.
            keras.utils.set_random_seed(1234)
            config = replace(self.request.config, use_skip=use_skip)
            sub = VisionRun(
                replace(self.request, config=config),
                on_epoch=self.on_epoch,
                on_message=self.on_message,
            )
            self._current = sub
            outcome = sub.run(label=label)
            self._current = None
            arms.append(outcome)
            if self.on_arm_done:
                self.on_arm_done(label, outcome)

        return {
            "arms": arms,
            "stopped": self._stop,
            "depth": self.request.config.depth,
        }


def epochs_to_reach(history: dict, target: float) -> int | None:
    """First epoch whose validation accuracy reached `target`, or None."""
    values = history.get("val_accuracy") or history.get("accuracy") or []
    for index, value in enumerate(values, start=1):
        if value >= target:
            return index
    return None


def format_comparison(result: dict) -> str:
    """A readout of the two arms that does not overstate what was measured."""
    arms = result.get("arms") or []
    if len(arms) < 2:
        return "Comparison incomplete."

    lines = [f"Same stack, {result['depth']} weight layers deep, identical init.\n"]
    for arm in arms:
        scores = arm.get("final_scores", {})
        lines.append(
            f"  {arm['label']:>9}: "
            f"val accuracy {scores.get('accuracy', float('nan')):.4f}  "
            f"val loss {scores.get('loss', float('nan')):.4f}  "
            f"({arm['trainable_params']:,} params, {arm['epochs_run']} epochs)"
        )

    residual = next((a for a in arms if a["label"] == "residual"), None)
    plain = next((a for a in arms if a["label"] == "plain"), None)
    if not (residual and plain):
        return "\n".join(lines)

    final_gap = (
        residual.get("final_scores", {}).get("accuracy", 0.0)
        - plain.get("final_scores", {}).get("accuracy", 0.0)
    ) * 100
    extra = residual["trainable_params"] - plain["trainable_params"]
    lines.append("")
    lines.append(f"  Final gap: {final_gap:+.1f} accuracy points for the shortcut.")

    # Speed is usually the real difference at CPU-reachable depths, so measure
    # it instead of asserting anything about degradation.
    best = max(
        max(a["history"].get("val_accuracy") or [0.0]) for a in (residual, plain)
    )
    target = 0.9 * best
    fast_r = epochs_to_reach(residual["history"], target)
    fast_p = epochs_to_reach(plain["history"], target)
    lines.append(
        f"  Epochs to reach {target:.3f} val accuracy: "
        f"residual {fast_r if fast_r else 'never'}, "
        f"plain {fast_p if fast_p else 'never'}."
    )

    lines.append("")
    lines.append(
        f"  Same convolutions, same depth, same initial weights. The residual arm "
        f"carries {extra:,} extra parameters "
        f"({100.0 * extra / max(plain['trainable_params'], 1):.1f}% more) for the "
        "1x1 projections that reshape the shortcut on stride-2 stages - far too "
        "small to explain the gap."
    )
    # The decisive diagnostic is TRAINING accuracy. A deep plain stack that
    # cannot even fit its training set is failing to optimise, not overfitting -
    # and that distinction is the entire argument of the ResNet paper.
    train_r = (residual["history"].get("accuracy") or [0.0])[-1]
    train_p = (plain["history"].get("accuracy") or [0.0])[-1]
    loss_r = (residual["history"].get("loss") or [0.0])[-1]
    loss_p = (plain["history"].get("loss") or [0.0])[-1]
    lines.append("")
    lines.append(
        f"  Final TRAINING accuracy: residual {train_r:.4f} (loss {loss_r:.4f}), "
        f"plain {train_p:.4f} (loss {loss_p:.4f})."
    )

    degraded = train_p < train_r - 0.02
    if degraded:
        lines.append("")
        lines.append(
            "  THIS IS THE DEGRADATION PROBLEM. The plain stack cannot even fit "
            "its own training data, while the residual one memorises it. That is "
            "an OPTIMISATION failure, not overfitting - more depth made the "
            "network harder to train, not more prone to memorising."
        )
        lines.append(
            "  That is precisely the result ResNet was invented to fix, and it is "
            "why the answer was a shortcut rather than more regularisation."
        )
    else:
        lines.append("")
        lines.append(
            "  Both arms fit the training set, so nothing is degrading here: at "
            "this depth the shortcut is buying OPTIMISATION SPEED, not reachable "
            "accuracy. Given enough epochs the plain arm usually catches up - "
            "BatchNorm already fixed much of what made moderately deep plain "
            "networks fail."
        )
        lines.append(
            "  To see the real degradation, go deeper: stages 3, blocks per stage "
            "8, stem filters 8 gives 50 layers, where the plain arm stops being "
            "able to fit the data at all."
        )
    return "\n".join(lines)
