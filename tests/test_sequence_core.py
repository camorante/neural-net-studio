"""Core sequence suite: no display, no Qt, real training.

Section 8 is the one that matters. The four tasks are not four flavours of the
same thing - two of them are designed to be lost, and the suite asserts that
they are lost, because a workspace that only ever shows recurrence winning
teaches the wrong reflex.
"""
import time

import _bootstrap
import numpy as np

from nnstudio.core import sequences as sq
from nnstudio.core.sequence_trainer import (
    ArchSweepRun, OrderProbeRun, SequenceRequest, SequenceRun,
    format_order_probe, format_run, format_sweep,
)

print("=== 1. the four tasks generate the shapes they promise ===")
for task in sq.TASKS:
    bundle = sq.make_sequences(sq.SequenceSpec(task=task, n_sequences=400,
                                               length=16, seed=3))
    print(f"  {task:<6} {bundle.kind:<9} x{bundle.x_train.shape} "
          f"y{bundle.y_train.shape}  {bundle.describe()[:52]}")
    assert bundle.x_train.shape[1:] == (16, 1)
    assert len(bundle.x_train) + len(bundle.x_val) >= 390
    if bundle.kind == sq.CLASSIFY:
        assert bundle.y_train.shape[1] == 2
        assert bundle.n_outputs == 2
    else:
        assert bundle.y_train.shape[1] == 1
        assert bundle.series is not None

print("\n=== 2. the forecast split is chronological, not random ===")
bundle = sq.make_sequences(sq.SequenceSpec(task=sq.SINE, n_sequences=400,
                                           length=16, seed=3))
# Every validation window must come from after every training window, which for
# a sliding window means the last training target appears inside the first
# validation window. A random split would break this immediately.
first_val_window = bundle.x_val[0, :, 0]
assert np.isclose(bundle.y_train[-1, 0], first_val_window[-1]), \
    "validation does not continue where training stopped - the split leaked"
print("  last training target is the newest step of the first validation window")

print("\n=== 3. shuffling keeps the values and destroys the order ===")
rng = np.random.default_rng(0)
original = bundle.x_train[:50]
scrambled = sq.shuffle_timesteps(original, rng)
assert np.allclose(np.sort(original, axis=1), np.sort(scrambled, axis=1)), \
    "shuffling changed the values, not just their order"
moved = float(np.mean(np.any(original != scrambled, axis=1)))
print(f"  same multiset of values: yes | sequences actually reordered: {moved:.0%}")
assert moved > 0.9
# Each sequence must get its own permutation, or the shuffle is just relabelling
# time and anything learnable before is learnable after.
first_two = [np.argsort(scrambled[i, :, 0]).tolist() for i in range(2)]
assert first_two[0] != first_two[1], "every sequence got the same permutation"

print("\n=== 4. the floors are the right kind for the task ===")
for task in sq.TASKS:
    data = sq.make_sequences(sq.SequenceSpec(task=task, n_sequences=400,
                                             length=16, seed=3))
    floor = sq.baseline(data)
    print(f"  {task:<6} {floor['name']:<14} {floor['value']:.5f}  "
          f"({'lower' if floor['better_is_lower'] else 'higher'} wins)")
    if data.kind == sq.FORECAST:
        assert floor["name"] == "persistence" and floor["better_is_lower"]
    else:
        assert floor["name"] == "majority class" and not floor["better_is_lower"]
        assert 0.3 <= floor["value"] <= 0.7, "a two-class floor should be near half"

print("\n=== 5. every architecture builds and is shaped right ===")
for kind in sq.KINDS:
    config = sq.SequenceConfig(length=16, kind=kind, units=8,
                               task_kind=sq.CLASSIFY, n_outputs=2)
    model = sq.build_sequence_model(config)
    trainable, _ = sq.count_params(model)
    print(f"  {kind:<7} out {tuple(model.output_shape)} "
          f"params {trainable:,} est {sq.estimate_params(config):,}")
    assert tuple(model.output_shape) == (None, 2)
    assert config.reads_order == (kind in (sq.LSTM, sq.GRU, sq.RNN))

print("\n=== 6. validate() refuses what cannot work ===")
bad = sq.SequenceConfig(length=16, kind=sq.CONV1D, bidirectional=True)
problems = sq.validate(bad)
print("  bidirectional conv1d:", problems[0][:76])
assert any("no direction" in p for p in problems)
short = sq.SequenceConfig(length=4, kind=sq.CONV1D)
print("  window shorter than the kernel:", sq.validate(short)[0][:76])
assert any("kernel" in p for p in sq.validate(short))

print("\n=== 7. a real run reports against its floor ===")
data = sq.make_sequences(sq.SequenceSpec(task=sq.ORDER, n_sequences=600,
                                         length=16, seed=3))
config = sq.SequenceConfig(length=16, kind=sq.LSTM, units=16,
                           task_kind=data.kind, n_outputs=data.n_outputs)
seen = []
started = time.monotonic()
result = SequenceRun(
    SequenceRequest(data=data, config=config, epochs=8, batch_size=32),
    on_epoch=lambda e, logs, label: seen.append(e),
    on_message=lambda m: None,
).run()
print(f"  epochs: {len(seen)} in {time.monotonic()-started:.1f}s | "
      f"accuracy {result['final_scores']['accuracy']:.3f} | "
      f"floor {result['baseline']:.3f}")
assert len(seen) == 8
assert "accuracy" in result["final_scores"], result["final_scores"]
assert result["metric"] == "accuracy" and not result["better_is_lower"]
print("  --- readout ---")
print(format_run(result))

print("\n=== 8. the two tasks that are meant to be lost, are lost ===")
# A random walk cannot be beaten by any model, and the sweep must say so rather
# than blaming a shortage of epochs.
walk = sq.make_sequences(sq.SequenceSpec(task=sq.WALK, n_sequences=600,
                                         length=16, seed=7))
walk_config = sq.SequenceConfig(length=16, kind=sq.LSTM, units=16,
                                task_kind=walk.kind, n_outputs=1)
sweep = ArchSweepRun(
    SequenceRequest(data=walk, config=walk_config, epochs=6, batch_size=32),
    kinds=(sq.LSTM, sq.DENSE), on_message=lambda m: None,
).run()
text = format_sweep(sweep)
print(text)
assert "NOBODY BEAT PERSISTENCE" in text, text
assert "Run it again" not in text, "more epochs will not help on a random walk"

print("\n=== 9. the order probe answers the question it was built for ===")
for task, expect in ((sq.ORDER, "ORDER MATTERS HERE"),
                     (sq.COUNT, "ORDER CARRIED NOTHING")):
    data = sq.make_sequences(sq.SequenceSpec(task=task, n_sequences=800,
                                             length=16, seed=7))
    config = sq.SequenceConfig(length=16, kind=sq.LSTM, units=16,
                               task_kind=data.kind, n_outputs=data.n_outputs)
    started = time.monotonic()
    probe = OrderProbeRun(
        SequenceRequest(data=data, config=config, epochs=12, batch_size=32),
        on_message=lambda m: None,
    ).run()
    text = format_order_probe(probe)
    real, shuffled = probe["arms"]
    print(f"\n  {task}: real {real['final_scores']['accuracy']:.3f} vs "
          f"shuffled {shuffled['final_scores']['accuracy']:.3f} "
          f"({time.monotonic()-started:.0f}s)")
    print("  verdict:", [l for l in text.splitlines() if expect in l][:1] or text)
    assert expect in text, text

print("\n=== 10. a sweep needs a real comparison ===")
try:
    ArchSweepRun(SequenceRequest(data=data, config=config), kinds=(sq.LSTM, sq.LSTM))
    raise AssertionError("one distinct architecture should not be a sweep")
except ValueError as exc:
    print("  refused:", exc)

print("\n=== 11. core stays Qt-free ===")
dirty = [str(p) for p in (_bootstrap.ROOT / "nnstudio" / "core").rglob("*.py")
         if "PyQt6" in p.read_text(encoding="utf-8")]
print(" ", dirty or "clean")
assert not dirty

print("\nSEQUENCE CORE OK")
