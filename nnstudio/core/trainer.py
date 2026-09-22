"""Training run, free of any UI framework.

The caller passes plain callables; a Qt worker adapts them to signals.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dataset import DataBundle
from .model_builder import NetworkConfig, build_model, keras_module, summary_text


@dataclass
class TrainingRequest:
    config: NetworkConfig
    data: DataBundle
    epochs: int = 50
    batch_size: int = 32
    shuffle: bool = True
    early_stopping: bool = False
    patience: int = 10


class TrainingRun:
    """Builds the model, trains it, and reports progress through callbacks."""

    def __init__(self, request: TrainingRequest, on_epoch=None, on_message=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def run(self) -> dict:
        request = self.request
        data = request.data

        self._say("Loading TensorFlow...")
        keras = keras_module()

        self._say("Building the model...")
        model = build_model(request.config)
        self._say(summary_text(model))

        run = self

        class _Bridge(keras.callbacks.Callback):
            """Bridges Keras epochs to the caller and honours the stop flag."""

            def on_epoch_end(self, epoch, logs=None):
                if run.on_epoch:
                    run.on_epoch(epoch + 1, {k: float(v) for k, v in (logs or {}).items()})
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

        self._say(f"Training on {data.n_train} rows, validating on {data.n_val}.")
        history = model.fit(
            data.x_train,
            data.y_train,
            validation_data=validation,
            epochs=max(1, request.epochs),
            batch_size=max(1, request.batch_size),
            shuffle=request.shuffle,
            verbose=0,
            callbacks=callbacks,
        )

        result = {
            "model": model,
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "stopped": self._stop,
            "epochs_run": len(history.history.get("loss", [])),
        }

        if data.n_val > 0:
            scores = model.evaluate(data.x_val, data.y_val, verbose=0, return_dict=True)
            result["final_scores"] = {k: float(v) for k, v in scores.items()}
        return result
