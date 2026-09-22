"""Training runs for the sequence workspace, free of any UI framework.

Three kinds of run live here, and only the first is ordinary:

  SequenceRun     one model, one dataset
  ArchSweepRun    the same data through LSTM, GRU, Conv1D and a plain Dense net,
                  from the same seed, so the only variable is the architecture
  OrderProbeRun   the same model twice - once on the real data, once with every
                  sequence's timesteps shuffled

The probe is the one worth the electricity. Everything else tells you how well a
model did; the probe tells you whether the problem ever needed a sequence model
at all, and that question has to be settled before the others mean anything.

Every readout here refuses to draw a conclusion it did not earn. A run that has
not beaten its give-up floor says so instead of reporting a number, and a probe
whose arms both sit on the floor reports nothing at all - two models that failed
equally have not demonstrated that order is irrelevant.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .model_builder import keras_module
from .sequences import (
    CLASSIFY,
    CONV1D,
    DENSE,
    FORECAST,
    GRU,
    KIND_LABELS,
    LSTM,
    SequenceBundle,
    SequenceConfig,
    TASK_NOTES,
    WALK,
    baseline,
    build_sequence_model,
    count_params,
    shuffle_timesteps,
)

# Every arm of every comparison starts from the same weights, so architecture -
# or the shuffle - is the only thing that differs.
ARM_SEED = 1234

# The four arms of the architecture sweep. Dense is not filler: it is the arm
# that says what the task is worth without any sequence machinery at all.
SWEEP_KINDS = (LSTM, GRU, CONV1D, DENSE)

# Measured on this project's own tasks (see the table in AGENTS.md). Fewer
# epochs than this and the arms have not separated, so the sweep would report a
# tie that is really a shortage of training.
SWEEP_EPOCHS = 20
SWEEP_BATCH = 32
PROBE_EPOCHS = 20

# A forecast has to beat persistence by more than this to count as a win, and a
# classifier has to beat the majority class by more than MEANINGFUL_POINTS.
MEANINGFUL_GAIN = 0.10          # 10% lower MSE
MEANINGFUL_POINTS = 0.05        # 5 accuracy points

# Below this the two probe arms have not separated and the probe says nothing.
ORDER_GAP_MSE = 0.10
ORDER_GAP_POINTS = 0.05


@dataclass
class SequenceRequest:
    """One supervised training run over a sequence dataset."""

    data: SequenceBundle
    config: SequenceConfig
    epochs: int = 20
    batch_size: int = 32
    early_stopping: bool = False
    patience: int = 6


class SequenceRun:
    """Builds a sequence model, trains it, reports back."""

    def __init__(self, request: SequenceRequest, on_epoch=None, on_message=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def run(self, model=None, label: str = "", data: SequenceBundle | None = None) -> dict:
        request = self.request
        config = request.config
        bundle = data if data is not None else request.data

        self._say("Loading TensorFlow...")
        keras = keras_module()

        if model is None:
            self._say(f"Building {config.describe()}.")
            model = build_sequence_model(config)

        trainable, _ = count_params(model)
        self._say(f"{model.name}: {trainable:,} trainable parameters.")
        if not config.reads_order:
            self._say(
                f"Note: {config.kind} cannot tell 'before' from 'after' across "
                "the window. Whatever it scores is what this task is worth "
                "without memory."
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
        if len(bundle.x_val) > 0:
            validation = (bundle.x_val, bundle.y_val)
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
            f"Training on {len(bundle.x_train)} windows, validating on "
            f"{len(bundle.x_val)}."
        )
        history = model.fit(
            bundle.x_train, bundle.y_train,
            validation_data=validation,
            epochs=int(request.epochs),
            batch_size=int(request.batch_size),
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )

        scores = {}
        if validation is not None:
            values = model.evaluate(bundle.x_val, bundle.y_val, verbose=0)
            values = values if isinstance(values, (list, tuple)) else [values]
            scores = {"loss": float(values[0])}
            if bundle.kind == CLASSIFY:
                # This Keras reports every compiled metric under the single
                # placeholder name "compile_metrics", so reading accuracy out of
                # metrics_names is a coin flip across versions. The number the
                # whole workspace is judged on should not depend on that, so it
                # is counted here from the predictions themselves.
                guess = np.argmax(model.predict(bundle.x_val, verbose=0), axis=1)
                truth = np.argmax(bundle.y_val, axis=1)
                scores["accuracy"] = float(np.mean(guess == truth))
            elif len(values) > 1:
                scores["mae"] = float(values[1])

        floor = baseline(bundle)
        return {
            "model": model,
            "history": {k: [float(v) for v in vs]
                        for k, vs in history.history.items()},
            "final_scores": scores,
            "epochs_run": len(history.history.get("loss", [])),
            "trainable_params": trainable,
            "kind": config.kind,
            "task": bundle.task,
            "task_kind": bundle.kind,
            "shuffled": bundle.shuffled,
            "baseline": floor["value"],
            "baseline_name": floor["name"],
            "metric": floor["metric"],
            "better_is_lower": floor["better_is_lower"],
            "label": label or KIND_LABELS.get(config.kind, config.kind),
        }


# ------------------------------------------------------------------- comparing

def _seeded(keras, seed: int = ARM_SEED) -> None:
    keras.utils.set_random_seed(int(seed))


class ArchSweepRun:
    """The same data through several architectures, from identical weights."""

    def __init__(self, request: SequenceRequest, kinds=None, on_epoch=None,
                 on_message=None, on_arm_done=None):
        self.request = request
        self.kinds = tuple(kinds or SWEEP_KINDS)
        if len(set(self.kinds)) < 2:
            raise ValueError("A sweep needs at least two different architectures")
        self.on_epoch = on_epoch
        self.on_message = on_message
        self.on_arm_done = on_arm_done
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> dict:
        keras = keras_module()
        arms = []
        for kind in self.kinds:
            if self._stop:
                break
            _seeded(keras)
            config = replace(self.request.config, kind=kind)
            if config.bidirectional and not config.reads_order:
                config = replace(config, bidirectional=False)
            label = kind
            if self.on_message:
                self.on_message(f"\n--- {KIND_LABELS.get(kind, kind)} ---")
            run = SequenceRun(replace(self.request, config=config),
                              on_epoch=self.on_epoch, on_message=self.on_message)
            if self._stop:
                break
            run._stop = self._stop
            outcome = run.run(label=label)
            outcome["label"] = label
            arms.append(outcome)
            if self.on_arm_done:
                self.on_arm_done(label, outcome)
        return {"arms": arms, "task": self.request.data.task,
                "task_kind": self.request.data.kind}


class OrderProbeRun:
    """The same model twice: real order, then with the timesteps shuffled.

    This is the only experiment here that can tell you a recurrent layer was
    pointless. Both arms see the same values, the same counts and the same
    number of weights; the shuffled arm simply cannot know what came first.
    """

    def __init__(self, request: SequenceRequest, on_epoch=None, on_message=None,
                 on_arm_done=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self.on_arm_done = on_arm_done
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> dict:
        keras = keras_module()
        bundle = self.request.data
        rng = np.random.default_rng(ARM_SEED)
        scrambled = SequenceBundle(
            x_train=shuffle_timesteps(bundle.x_train, rng),
            y_train=bundle.y_train,
            x_val=shuffle_timesteps(bundle.x_val, rng),
            y_val=bundle.y_val,
            name=bundle.name + " (timesteps shuffled)",
            task=bundle.task, kind=bundle.kind, length=bundle.length,
            n_features=bundle.n_features, class_names=bundle.class_names,
            shuffled=True, series=None,
        )

        arms = []
        for label, data in (("real order", bundle), ("time shuffled", scrambled)):
            if self._stop:
                break
            _seeded(keras)
            if self.on_message:
                self.on_message(f"\n--- {label} ---")
            run = SequenceRun(self.request, on_epoch=self.on_epoch,
                              on_message=self.on_message)
            run._stop = self._stop
            outcome = run.run(label=label, data=data)
            outcome["label"] = label
            arms.append(outcome)
            if self.on_arm_done:
                self.on_arm_done(label, outcome)
        return {"arms": arms, "task": bundle.task, "task_kind": bundle.kind,
                "probe": True}


# ------------------------------------------------------------------- readouts

def _score(outcome: dict) -> float:
    """The number this task is judged on: val MSE, or val accuracy."""
    scores = outcome.get("final_scores") or {}
    if outcome.get("task_kind") == CLASSIFY:
        return float(scores.get("accuracy", 0.0))
    return float(scores.get("loss", float("nan")))


def _beat_floor(outcome: dict) -> bool:
    score, floor = _score(outcome), float(outcome.get("baseline", float("nan")))
    if score != score or floor != floor:
        return False
    if outcome.get("better_is_lower"):
        return score < floor * (1.0 - MEANINGFUL_GAIN)
    return score > floor + MEANINGFUL_POINTS


def format_run(outcome: dict) -> str:
    """One run, said against its floor - never on its own."""
    score, floor = _score(outcome), float(outcome.get("baseline", float("nan")))
    lower = bool(outcome.get("better_is_lower"))
    unit = "val MSE" if lower else "val accuracy"
    lines = [
        f"{KIND_LABELS.get(outcome.get('kind'), outcome.get('kind'))}",
        f"  {unit:<16}{score:.5f}   ({outcome.get('trainable_params', 0):,} "
        f"parameters, {outcome.get('epochs_run', 0)} epochs)",
        f"  {outcome.get('baseline_name', 'giving up'):<16}{floor:.5f}   "
        f"({'lower is better' if lower else 'higher is better'})",
        "",
    ]

    if not _beat_floor(outcome):
        if lower:
            lines.append(
                "  THIS RUN DID NOT BEAT PERSISTENCE. Answering that the next "
                "value equals the last one scores at least as well as the model "
                "you just trained, so nothing has been learned yet."
            )
        else:
            lines.append(
                "  THIS RUN DID NOT BEAT THE MAJORITY CLASS. Always answering "
                "with the commonest label scores at least as well, so nothing "
                "has been learned yet."
            )
        lines.append(
            "  Give it more epochs, or a task where the past actually predicts "
            "the future - the two are not the same problem and the readout "
            "cannot tell them apart for you."
        )
    elif lower:
        lines.append(
            f"  Beat persistence: {floor / max(score, 1e-9):.2f}x lower error. "
            "The past carries information the last value alone does not."
        )
    else:
        lines.append(
            f"  Beat the majority class by {100 * (score - floor):.1f} points."
        )

    if outcome.get("task") == WALK and _beat_floor(outcome):
        lines.append("")
        lines.append(
            "  TREAT THIS WIN WITH SUSPICION. On a random walk the best "
            "possible prediction of the next value IS the last value - that is "
            "what a random walk means. A model that beats persistence here has "
            "found structure in the validation noise, not in the series. Change "
            "the seed and watch it evaporate."
        )
    if outcome.get("shuffled"):
        lines.append("")
        lines.append(
            "  This run used SHUFFLED timesteps. Whatever it scored, it scored "
            "without knowing what came first."
        )
    return "\n".join(lines)


def format_sweep(outcome: dict) -> str:
    """Four architectures on one task, ordered by what they actually scored."""
    arms = outcome.get("arms") or []
    if len(arms) < 2:
        return "Not enough arms finished to compare."

    lower = bool(arms[0].get("better_is_lower"))
    floor = float(arms[0].get("baseline", float("nan")))
    unit = "val MSE" if lower else "val accuracy"
    lines = [
        "Same windows, same initial weights, same number of epochs. Only the "
        "architecture changed.",
        "",
    ]
    for arm in arms:
        lines.append(
            f"  {arm['label']:<8} {unit} {_score(arm):.5f}   "
            f"({arm.get('trainable_params', 0):,} params, "
            f"{arm.get('epochs_run', 0)} epochs)"
        )
    lines.append(
        f"  {arm.get('baseline_name', 'floor'):<8} {unit} {floor:.5f}   "
        "(no model at all)"
    )
    lines.append("")

    winners = [a for a in arms if _beat_floor(a)]
    if not winners:
        if outcome.get("task") == WALK:
            # Losing here is not a shortage of training. It is the result.
            lines.append(
                "  NOBODY BEAT PERSISTENCE, AND NOBODY WAS GOING TO. On a "
                "random walk the next value is the last value plus fresh noise, "
                "so no function of the past can do better than repeating the "
                "last value. This is the one task in this workspace where the "
                "correct outcome is that every architecture loses."
            )
            lines.append(
                "  More epochs will not fix it, and neither will a bigger "
                "model. Read the gap as what it is: the price of the noise."
            )
            return "\n".join(lines)
        lines.append(
            "  NO CONCLUSION IS AVAILABLE FROM THIS RUN. Not one architecture "
            f"beat {arms[0].get('baseline_name', 'the floor')}, so they are not "
            "tied - none of them has started learning. This says nothing "
            "whatsoever about which architecture suits the task."
        )
        lines.append(
            f"  Run it again with about {SWEEP_EPOCHS} epochs before reading "
            "anything into the order."
        )
        return "\n".join(lines)

    best = min(winners, key=_score) if lower else max(winners, key=_score)
    memory = [a for a in winners if a["label"] in (LSTM, GRU)]
    blind = [a for a in arms if a["label"] in (CONV1D, DENSE)]
    lines.append(f"  Best: {KIND_LABELS.get(best['label'], best['label'])}")

    if blind and memory:
        best_blind = min(blind, key=_score) if lower else max(blind, key=_score)
        best_memory = min(memory, key=_score) if lower else max(memory, key=_score)
        gap_matters = (
            _score(best_memory) < _score(best_blind) * (1.0 - MEANINGFUL_GAIN)
            if lower else
            _score(best_memory) > _score(best_blind) + MEANINGFUL_POINTS
        )
        if gap_matters:
            lines.append(
                f"  The recurrence earned its place: {best_memory['label']} beat "
                f"the best arm without it ({best_blind['label']}) by a clear "
                "margin, at the same epochs and from the same weights."
            )
        elif _beat_floor(best_blind):
            cost = best_memory.get("trainable_params", 0)
            cheap = best_blind.get("trainable_params", 0)
            lines.append(
                f"  The recurrence bought NOTHING here: {best_blind['label']} "
                f"matched it with {cheap:,} parameters against {cost:,}."
            )
            lines.append(
                "  Be careful what you conclude from that. A Dense net on a "
                "fixed window is not order-blind - it gets one input per "
                "timestep and can compare any two of them directly. What it "
                "cannot do is reuse anything it learns from one position at "
                "another, or survive a change of window length at all. So this "
                "says the task is easy at this length, not that order is "
                "irrelevant. Only the shuffle probe can tell you that."
            )
    return "\n".join(lines)


def format_order_probe(outcome: dict) -> str:
    """Did order carry anything? The one question worth asking first."""
    arms = outcome.get("arms") or []
    if len(arms) < 2:
        return "The probe needs both arms to finish."

    real, shuffled = arms[0], arms[1]
    lower = bool(real.get("better_is_lower"))
    unit = "val MSE" if lower else "val accuracy"
    floor = float(real.get("baseline", float("nan")))
    lines = [
        "The same model, twice. Same weights to start, same values in every "
        "window, same counts. The second one had every sequence's timesteps "
        "permuted, so it could not know what came first.",
        "",
        f"  real order     {unit} {_score(real):.5f}",
        f"  time shuffled  {unit} {_score(shuffled):.5f}",
        f"  {real.get('baseline_name', 'floor'):<14} {unit} {floor:.5f}",
        "",
    ]

    if not _beat_floor(real) and not _beat_floor(shuffled):
        lines.append(
            "  NO CONCLUSION IS AVAILABLE. Neither arm beat "
            f"{real.get('baseline_name', 'the floor')}, so they have not been "
            "shown to be equal - they have both failed, which is a different "
            "thing and proves nothing about order."
        )
        if outcome.get("task") == WALK:
            lines.append(
                "  On this task that is expected and permanent: persistence is "
                "the best any model can do on a random walk, so the probe has "
                "no room to measure anything. Try it on a task a model can "
                "actually win."
            )
        else:
            lines.append(
                f"  Train for longer - around {PROBE_EPOCHS} epochs on these "
                "tasks - and probe again."
            )
        return "\n".join(lines)

    separated = (
        _score(shuffled) > _score(real) * (1.0 + ORDER_GAP_MSE) if lower
        else _score(real) > _score(shuffled) + ORDER_GAP_POINTS
    )
    if separated:
        if lower:
            damage = f"{_score(shuffled) / max(_score(real), 1e-9):.2f}x worse"
        else:
            damage = f"{100 * (_score(real) - _score(shuffled)):.1f} points worse"
        lines.append(
            f"  ORDER MATTERS HERE. Destroying it made the model {damage}. "
            "That gap is the part of the answer that lived in the sequence "
            "rather than in the values, and it is the only thing a recurrent "
            "layer can buy you."
        )
    else:
        lines.append(
            "  ORDER CARRIED NOTHING. The shuffled arm did as well as the real "
            "one, so everything the model learned it could learn from an "
            "unordered pile of values. A recurrent layer on this task is cost "
            "with no return - a Dense network would do the same job faster."
        )
        lines.append(
            "  That is not a failure of the model. It is a fact about the task, "
            "and it is worth knowing before you spend a week tuning an LSTM."
        )

    note = TASK_NOTES.get(outcome.get("task"))
    if note:
        lines.append("")
        lines.append(f"  About this task: {note}")
    return "\n".join(lines)
