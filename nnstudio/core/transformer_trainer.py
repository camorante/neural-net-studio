"""Training runs for the transformer workspace, free of any UI framework.

Two runs and three readouts. The runs are ordinary; the readouts are where this
workspace either teaches something true or something flattering.

  TransformerRun     one model, one dataset, and a faithfulness test of its map
  PositionProbeRun   the same model twice, with and without positional encoding

`format_faithfulness()` is the one that matters. Measured on this project's own
tasks, a model with its positions removed produced a map on the `match` task
that looked MORE focused than the correct map on `majority` - while failing the
task, and while erasing the token it "attended to" changed nothing at all. A
heatmap is an image of weights, not a report of reasons, and the readout has to
say which one it is looking at.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .model_builder import keras_module
from .transformer import (
    TASK_NEEDS_POSITIONS,
    TASK_NOTES,
    TokenBundle,
    TransformerConfig,
    accuracy,
    baseline,
    build_transformer,
    count_params,
    faithfulness,
)

ARM_SEED = 1234

# Measured over four seeds with the default model (two blocks, 64 wide): all
# three tasks are solved at 40 epochs whenever they can be solved at all.
DEFAULT_EPOCHS = 40
PROBE_EPOCHS = 40

# A classifier has to beat the majority class by more than this to have learned.
MEANINGFUL_POINTS = 0.05

# Faithfulness thresholds, set from the measurements rather than from taste.
# Faithful maps dropped accuracy by 0.94 to 1.00 when their attended token was
# erased, against 0.00 to 0.06 for the control; decorative ones by under 0.02.
FAITHFUL_DROP = 0.20
FAITHFUL_RATIO = 3.0
SMALL_DROP = 0.05
SOLVED = 0.90

# The two probe arms must differ by more than this before position is credited.
POSITION_GAP = 0.10


@dataclass
class TransformerRequest:
    """One training run over a token dataset."""

    data: TokenBundle
    config: TransformerConfig
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = 32
    early_stopping: bool = False
    patience: int = 6


class TransformerRun:
    """Builds the transformer, trains it, and questions its attention map."""

    def __init__(self, request: TransformerRequest, on_epoch=None, on_message=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self.on_message:
            self.on_message(message)

    def run(self, label: str = "", config: TransformerConfig | None = None) -> dict:
        request = self.request
        config = config or request.config
        bundle = request.data

        self._say("Loading TensorFlow...")
        keras = keras_module()

        self._say(f"Building a transformer: {config.describe()}.")
        if not config.positional:
            self._say(
                "Positional encoding is OFF. Self-attention treats its input "
                "as an unordered set, so this model cannot tell which token came "
                "first - whatever it scores, it scores without that."
            )
        model, attention_model = build_transformer(config)
        trainable, _ = count_params(model)
        self._say(f"{model.name}: {trainable:,} trainable parameters.")

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
        if request.early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=max(1, request.patience),
                restore_best_weights=True, verbose=0,
            ))

        self._say(f"Training on {len(bundle.x_train)} sequences, validating on "
                  f"{len(bundle.x_val)}.")
        history = model.fit(
            bundle.x_train, bundle.y_train,
            validation_data=(bundle.x_val, bundle.y_val),
            epochs=int(request.epochs), batch_size=int(request.batch_size),
            callbacks=callbacks, shuffle=True, verbose=0,
        )

        # Counted from the predictions: this Keras names every compiled metric
        # "compile_metrics", so reading accuracy by name is not reliable.
        score = accuracy(model, bundle.x_val, bundle.y_val)
        floor = baseline(bundle)

        self._say("Testing the attention map: erasing the token it weighed most, "
                  "and a random other token as a control...")
        faith = faithfulness(model, attention_model, bundle)

        return {
            "model": model,
            "attention_model": attention_model,
            "history": {k: [float(v) for v in vs]
                        for k, vs in history.history.items()},
            "final_scores": {"accuracy": score},
            "epochs_run": len(history.history.get("loss", [])),
            "trainable_params": trainable,
            "positional": bool(config.positional),
            "n_blocks": int(config.n_blocks),
            "task": bundle.task,
            "baseline": floor["value"],
            "baseline_name": floor["name"],
            "faithfulness": faith,
            "label": label or ("with positions" if config.positional
                               else "no positions"),
        }


class PositionProbeRun:
    """The same model twice: with positional encoding, then without it.

    Everything else is identical, down to the initial weights of every layer the
    two arms share. So the gap between them is exactly what knowing the order
    was worth on this task.
    """

    def __init__(self, request: TransformerRequest, on_epoch=None,
                 on_message=None, on_arm_done=None):
        self.request = request
        self.on_epoch = on_epoch
        self.on_message = on_message
        self.on_arm_done = on_arm_done
        self._stop = False
        self._current = None

    def stop(self) -> None:
        self._stop = True
        if self._current:
            self._current.stop()

    def run(self) -> dict:
        keras = keras_module()
        arms = []
        for label, positional in (("with positions", True), ("no positions", False)):
            if self._stop:
                break
            keras.utils.set_random_seed(ARM_SEED)
            if self.on_message:
                self.on_message(f"\n--- {label} ---")
            self._current = TransformerRun(self.request, on_epoch=self.on_epoch,
                                           on_message=self.on_message)
            outcome = self._current.run(
                label=label,
                config=replace(self.request.config, positional=positional),
            )
            arms.append(outcome)
            if self.on_arm_done:
                self.on_arm_done(label, outcome)
        return {"arms": arms, "task": self.request.data.task, "probe": True}


# ------------------------------------------------------------------- readouts

def _score(outcome: dict) -> float:
    return float((outcome.get("final_scores") or {}).get("accuracy", 0.0))


def _beat_floor(outcome: dict) -> bool:
    return _score(outcome) > float(outcome.get("baseline", 1.0)) + MEANINGFUL_POINTS


def verdict(outcome: dict) -> str:
    """Classify the attention map: faithful, distributed, decorative, or moot.

    Kept separate from the prose so the workspace can show the one-word answer
    on a label and the tests can assert it without matching sentences.
    """
    faith = outcome.get("faithfulness") or {}
    if not _beat_floor(outcome):
        return "moot"
    hit = float(faith.get("attended_drop", 0.0))
    control = max(float(faith.get("control_drop", 0.0)), 0.02)
    if hit >= FAITHFUL_DROP and hit >= FAITHFUL_RATIO * control:
        return "faithful"
    if hit < SMALL_DROP and _score(outcome) >= SOLVED:
        return "distributed"
    return "decorative"


def format_faithfulness(outcome: dict) -> str:
    """What erasing the attended token actually did - the map's only real test."""
    faith = outcome.get("faithfulness") or {}
    if not faith:
        return "No attention map was tested."
    kind = verdict(outcome)
    lines = [
        f"Attention map, block {int(faith.get('block', 0)) + 1} (the one nearest "
        f"the answer), tested on {faith.get('n', 0)} validation sequences:",
        f"  accuracy, untouched                 {faith['intact']:.3f}",
        # Printed as the change, with its own sign: an erasure can nudge
        # accuracy UP by chance, and "--0.003" reads as a typo, not a result.
        f"  after erasing the ATTENDED token    {faith['attended_erased']:.3f}"
        f"   ({-faith['attended_drop']:+.3f})",
        f"  after erasing a RANDOM other token  {faith['control_erased']:.3f}"
        f"   ({-faith['control_drop']:+.3f})",
        f"  how focused the map looks           {faith['sharpness']:.2f}"
        "   (0 = flat, 1 = one column takes everything)",
        "",
    ]
    if kind == "moot":
        lines.append(
            "  NO CONCLUSION. The model has not beaten the majority class, so "
            "its attention map describes nothing it learned - there is nothing "
            "learned for it to describe."
        )
    elif kind == "faithful":
        # A ratio against a control of zero is a division, not a finding, so
        # say what happened instead of printing an artefact of the floor.
        if faith["control_drop"] < 0.02:
            comparison = ("while erasing an arbitrary one changed essentially "
                          "nothing")
        else:
            ratio = faith["attended_drop"] / faith["control_drop"]
            comparison = f"{ratio:.0f}x more than erasing an arbitrary one"
        lines.append(
            f"  THIS MAP EARNED ITS READING. Erasing the token it pointed at cost "
            f"{faith['attended_drop']:.3f} of accuracy, {comparison}. The answer "
            "really did depend on the position the map highlights."
        )
    elif kind == "distributed":
        lines.append(
            "  NO SINGLE TOKEN MATTERS, AND THE MAP IS RIGHT TO SAY SO. The model "
            "is correct, and erasing any one token - attended or not - changes "
            "nothing. The answer is spread across the whole sequence, so a flat "
            "map is the honest picture. A sharp one here would be the suspicious "
            "result."
        )
    else:
        lines.append(
            "  THIS MAP IS DECORATION. It points somewhere, but erasing the token "
            "it points at hurts no more than erasing a random one. Whatever the "
            "model is using, the heatmap is not showing you where it is."
        )
        if faith["sharpness"] >= 0.25:
            lines.append(
                f"  Note how focused it LOOKS ({faith['sharpness']:.2f}). That is the "
                "trap: a confident-looking heatmap is an image of weights, not a "
                "report of reasons."
            )
    return "\n".join(lines)


def format_run(outcome: dict) -> str:
    """One run, against its floor, followed by the test of its map."""
    score, floor = _score(outcome), float(outcome.get("baseline", 0.0))
    lines = [
        f"Transformer, {outcome.get('n_blocks', 1)} block(s), "
        + ("with" if outcome.get("positional") else "WITHOUT")
        + " positional encoding",
        f"  val accuracy    {score:.3f}   ({outcome.get('trainable_params', 0):,} "
        f"parameters, {outcome.get('epochs_run', 0)} epochs)",
        f"  majority class  {floor:.3f}",
        "",
    ]
    if not _beat_floor(outcome):
        lines.append(
            "  THIS RUN DID NOT BEAT THE MAJORITY CLASS. Nothing has been learned "
            "yet."
        )
    elif score < SOLVED and TASK_NEEDS_POSITIONS.get(outcome.get("task")) \
            and not outcome.get("positional"):
        lines.append(
            f"  Above the floor, but nowhere near solved. Without positions this "
            "task cannot be solved at all; the model is guessing the commonest "
            "token in each sequence, which is right about a quarter of the time "
            "only because the answer is one of the tokens being counted."
        )
    elif score < SOLVED:
        lines.append(
            f"  Beat the floor by {100 * (score - floor):.1f} points but has not "
            "solved the task. On `match` a model 32 wide got stuck at 0.70 on half "
            "the seeds for as long as it was trained - that is a plateau, not a "
            "shortage of epochs. Widen it, or try another seed."
        )
    else:
        lines.append(f"  Solved: {100 * (score - floor):.1f} points above the floor.")
    lines.append("")
    lines.append(format_faithfulness(outcome))
    return "\n".join(lines)


def format_position_probe(outcome: dict) -> str:
    """Was knowing the order worth anything on this task?"""
    arms = outcome.get("arms") or []
    if len(arms) < 2:
        return "The probe needs both arms to finish."
    real, blind = arms[0], arms[1]
    floor = float(real.get("baseline", 0.0))
    lines = [
        "The same transformer twice, from the same initial weights. The second "
        "one had its positional encoding removed, so self-attention saw each "
        "sequence as an unordered set of tokens.",
        "",
        f"  with positions  val accuracy {_score(real):.3f}",
        f"  no positions    val accuracy {_score(blind):.3f}",
        f"  majority class  val accuracy {floor:.3f}",
        "",
    ]
    if not _beat_floor(real) and not _beat_floor(blind):
        lines.append(
            "  NO CONCLUSION IS AVAILABLE. Neither arm beat the majority class, "
            "so they have not been shown to be equal - they have both failed. "
            f"Train for around {PROBE_EPOCHS} epochs and probe again."
        )
        return "\n".join(lines)

    gap = _score(real) - _score(blind)
    if gap > POSITION_GAP:
        lines.append(
            f"  POSITION IS ESSENTIAL HERE. Removing it cost {100 * gap:.1f} "
            "points. That is not the model being weaker: attention without "
            "positions is permutation-invariant, so 'first', 'after' and 'next "
            "to' do not exist for it. No amount of training changes that."
        )
    else:
        lines.append(
            "  POSITION CARRIED NOTHING. The model did as well without it, "
            "because the answer never depended on where any token was - only on "
            "which tokens were there."
        )
    note = TASK_NOTES.get(outcome.get("task"))
    if note:
        lines.append("")
        lines.append(f"  About this task: {note}")
    return "\n".join(lines)
