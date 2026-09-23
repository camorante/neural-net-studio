"""Core transformer suite: no display, no Qt, real training.

Section 7 is the reason this suite exists. The attention map gets one of four
verdicts, and each is asserted on a model that genuinely earns it - including
"decorative", for the positionless model whose map looks focused while the
answer it gives has nothing to do with where it points.
"""
import time

import _bootstrap
import numpy as np

from nnstudio.core import transformer as tf_
from nnstudio.core.transformer_trainer import (
    PositionProbeRun, TransformerRequest, TransformerRun,
    format_faithfulness, format_position_probe, format_run, verdict,
)

print("=== 1. the three tasks generate what they promise ===")
for task in tf_.TASKS:
    b = tf_.make_tokens(tf_.TokenSpec(task=task, n_sequences=400, length=10,
                                      vocab=6, seed=3))
    labels = np.argmax(b.y_train, axis=1) + 1          # back to token ids
    x = b.x_train
    if task == tf_.FIRST:
        assert (labels == x[:, 0]).all(), "label is not the first token"
    elif task == tf_.MATCH:
        cue = np.argmax(x == tf_.CUE, axis=1)
        assert ((x == tf_.CUE).sum(axis=1) == 1).all(), "exactly one cue per sequence"
        assert (cue < b.length - 1).all(), "a cue on the last step points at nothing"
        assert (labels == x[np.arange(len(x)), cue + 1]).all()
    else:
        assert all(np.bincount(row, minlength=6).argmax() == lab
                   for row, lab in zip(x, labels)), "label is not the majority"
        assert (x != tf_.CUE).all(), "the cue is reserved and never content"
    print(f"  {task:<9} x{x.shape} classes {b.n_outputs} | labels check out")
    assert b.n_outputs == 5

print("\n=== 2. the floor is the majority class, near chance ===")
b = tf_.make_tokens(tf_.TokenSpec(task=tf_.MATCH, seed=7))
floor = tf_.baseline(b)
print(f"  {floor['name']} {floor['value']:.3f} | chance {1 / b.n_outputs:.3f}")
assert floor["metric"] == "accuracy" and not floor["better_is_lower"]
assert 0.08 < floor["value"] < 0.25

print("\n=== 3. the model builds, and so does its attention probe ===")
cfg = tf_.TransformerConfig(length=12, vocab=8, n_outputs=7)
model, attn = tf_.build_transformer(cfg)
trainable, _ = tf_.count_params(model)
print(f"  params {trainable:,} | estimate {tf_.estimate_params(cfg):,}")
assert trainable == tf_.estimate_params(cfg)
maps = tf_.attention_maps(attn, b.x_val[:5])
print(f"  blocks {len(maps)} | map shape {maps[0].shape}")
assert len(maps) == cfg.n_blocks
assert maps[0].shape == (5, cfg.n_heads, 12, 12)
assert np.allclose(maps[0].sum(axis=-1), 1.0, atol=1e-4), "attention rows must sum to 1"
blind = tf_.TransformerConfig(length=12, vocab=8, n_outputs=7, positional=False)
print(f"  without positions: {tf_.estimate_params(blind):,} params "
      f"({tf_.estimate_params(cfg) - tf_.estimate_params(blind):,} fewer = 12 x 64)")
assert tf_.estimate_params(cfg) - tf_.estimate_params(blind) == 12 * 64

print("\n=== 4. validate() refuses what cannot work ===")
odd = tf_.TransformerConfig(d_model=30, n_heads=4)
print("  30 wide, 4 heads:", tf_.validate(odd)[0][:70])
assert any("divide" in p for p in tf_.validate(odd))

print("\n=== 5. sharpness reads a flat map as 0 and a spike as 1 ===")
flat = [np.full((2, 1, 6, 6), 1 / 6, dtype="float32")]
spike = np.zeros((2, 1, 6, 6), dtype="float32"); spike[..., 3] = 1.0
print(f"  flat {tf_.map_sharpness(flat):.2f} | spike {tf_.map_sharpness([spike]):.2f}")
assert tf_.map_sharpness(flat) < 0.01 and tf_.map_sharpness([spike]) > 0.99
assert (tf_.attended_positions([spike]) == 3).all()

print("\n=== 6. the position probe on `first` ===")
first = tf_.make_tokens(tf_.TokenSpec(task=tf_.FIRST, seed=7))
config = tf_.TransformerConfig(length=first.length, vocab=first.vocab,
                               n_outputs=first.n_outputs)
started = time.monotonic()
probe = PositionProbeRun(TransformerRequest(data=first, config=config),
                         on_message=lambda m: None).run()
with_pos, without = probe["arms"]
print(f"  with {with_pos['final_scores']['accuracy']:.3f} | without "
      f"{without['final_scores']['accuracy']:.3f} ({time.monotonic()-started:.0f}s)")
text = format_position_probe(probe)
assert "POSITION IS ESSENTIAL HERE" in text, text

print("\n=== 7. the four verdicts, each on a model that earns it ===")
print(f"  with positions  -> {verdict(with_pos)}")
assert verdict(with_pos) == "faithful", format_faithfulness(with_pos)
print(f"  no positions    -> {verdict(without)} "
      f"(map sharpness {without['faithfulness']['sharpness']:.2f})")
assert verdict(without) == "decorative", format_faithfulness(without)
assert "THIS MAP IS DECORATION" in format_faithfulness(without)

majority = tf_.make_tokens(tf_.TokenSpec(task=tf_.MAJORITY, seed=7))
config = tf_.TransformerConfig(length=majority.length, vocab=majority.vocab,
                               n_outputs=majority.n_outputs)
spread = TransformerRun(TransformerRequest(data=majority, config=config),
                        on_message=lambda m: None).run()
print(f"  majority task   -> {verdict(spread)} "
      f"(accuracy {spread['final_scores']['accuracy']:.3f})")
assert verdict(spread) == "distributed", format_faithfulness(spread)

match = tf_.make_tokens(tf_.TokenSpec(task=tf_.MATCH, seed=7))
config = tf_.TransformerConfig(length=match.length, vocab=match.vocab,
                               n_outputs=match.n_outputs)
untrained = TransformerRun(TransformerRequest(data=match, config=config, epochs=1),
                           on_message=lambda m: None).run()
print(f"  one epoch       -> {verdict(untrained)} "
      f"(accuracy {untrained['final_scores']['accuracy']:.3f})")
assert verdict(untrained) == "moot", format_faithfulness(untrained)
assert "NO CONCLUSION" in format_faithfulness(untrained)

print("\n  --- the faithful readout ---")
print(format_run(with_pos))
print("\n  --- the decorative readout ---")
print(format_faithfulness(without))
for text in (format_faithfulness(with_pos), format_faithfulness(without)):
    assert "--" not in text.replace("---", ""), "a double sign reads as a typo"
# A ratio over a zero control is an artefact of the floor, not a measurement.
if with_pos["faithfulness"]["control_drop"] < 0.02:
    assert "50x" not in format_faithfulness(with_pos)
    assert "changed essentially nothing" in format_faithfulness(with_pos)

print("\n=== 8. core stays Qt-free ===")
dirty = [str(p) for p in (_bootstrap.ROOT / "nnstudio" / "core").rglob("*.py")
         if "PyQt6" in p.read_text(encoding="utf-8")]
print(" ", dirty or "clean")
assert not dirty

print("\nTRANSFORMER CORE OK")
