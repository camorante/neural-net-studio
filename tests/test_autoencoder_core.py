"""Core autoencoder suite: no display, no Qt, real training.

Covers the framework-free half: configuration validation, geometry, both model
variants, an actual unsupervised run, reconstruction, noise, the anomaly
holdout, and the latent sweep. Ends by asserting that core stays Qt-free.
"""
import time

import _bootstrap
import numpy as np

from nnstudio.core import vision as vz
from nnstudio.core import autoencoder as ae
from nnstudio.core.autoencoder_trainer import (
    AutoencoderRequest, AutoencoderRun, LatentSweepRun,
    format_sweep, hardest_to_reconstruct,
)

print("=== 1. validate() refuses what cannot work ===")
ok = ae.AutoencoderConfig(input_shape=(32, 32, 3), latent_dim=16, stages=2)
assert ae.validate(ok) == [], ae.validate(ok)
print("  sane config      :", "clean")

no_squeeze = ae.AutoencoderConfig(input_shape=(32, 32, 3), latent_dim=4096)
problems = ae.validate(no_squeeze)
assert any("no bottleneck" in p for p in problems), problems
print("  latent >= inputs :", problems[0][:72])

odd = ae.AutoencoderConfig(input_shape=(48, 48, 3), stages=5, latent_dim=8)
problems = ae.validate(odd)
assert any("halved" in p for p in problems), problems
print("  indivisible size :", problems[0][:72])

print("\n=== 2. geometry and compression ===")
cfg = ae.AutoencoderConfig(input_shape=(32, 32, 3), latent_dim=16, stages=2,
                           base_filters=16)
print("  n_inputs       :", cfg.n_inputs, "(expect 3072)")
print("  compression    :", f"{cfg.compression:.0f}x", "(expect 192x)")
print("  bottleneck map :", cfg.bottleneck_map(), "(expect (8, 8, 32))")
assert cfg.n_inputs == 3072 and cfg.bottleneck_map() == (8, 8, 32)
assert abs(cfg.compression - 192.0) < 1e-6
stack = cfg.stack()
print("  diagram stack  :", " -> ".join(c["name"] for c in stack))
print("  values         :", [c["values"] for c in stack])
waist = min(c["values"] for c in stack)
assert waist == 16, waist
assert stack[0]["values"] == stack[-1]["values"] == 3072
# the hourglass must actually pinch in the middle, not somewhere else
assert stack[len(stack) // 2]["kind"] == "latent"

print("\n=== 3. both variants build and are shaped right ===")
for variant in ("conv", "dense"):
    config = ae.AutoencoderConfig(
        input_shape=(32, 32, 3), variant=variant, latent_dim=16, stages=2,
        base_filters=8, hidden_units=128,
    )
    model, encoder = ae.build_autoencoder(config)
    trainable, total = ae.count_params(model)
    print(f"  {variant:>5}: out {tuple(model.output_shape)} "
          f"latent {tuple(encoder.output_shape)} "
          f"params {trainable:,} est {ae.estimate_params(config):,}")
    assert tuple(model.output_shape) == (None, 32, 32, 3)
    assert tuple(encoder.output_shape) == (None, 16)
    assert model.loss == "mse"

print("\n=== 4. a real (tiny) training run, unsupervised ===")
bundle = vz.make_synthetic(vz.SyntheticSpec(n_images=240, image_size=32,
                                            n_classes=3, noise=0.05, seed=3))
print(" ", bundle.describe())
config = ae.AutoencoderConfig(input_shape=bundle.input_shape, latent_dim=16,
                              stages=2, base_filters=8)
seen = []
run = AutoencoderRun(
    AutoencoderRequest(data=bundle, config=config, epochs=3, batch_size=32),
    on_epoch=lambda e, logs, label: seen.append((e, logs)),
    on_message=lambda m: None,
)
started = time.monotonic()
result = run.run()
print(f"  epochs reported : {len(seen)} (expect 3)  in {time.monotonic()-started:.1f}s")
assert len(seen) == 3
print("  history keys    :", sorted(result["history"]))
assert "val_loss" in result["history"] and "val_mae" in result["history"]
print("  final val MSE   :", f"{result['final_scores']['loss']:.5f}")
print("  loss fell       :", result["history"]["loss"][0], "->",
      result["history"]["loss"][-1])
assert result["history"]["loss"][-1] < result["history"]["loss"][0]
assert result["latent_dim"] == 16
assert "anomaly" not in result, "no class was held out, so there must be no anomaly block"

print("\n=== 5. reconstruction comes back as displayable pixels ===")
model = result["model"]
sample = bundle.x_val[:8]
rebuilt = ae.reconstruct(model, sample)
print("  input  range :", f"{sample.min():.1f} .. {sample.max():.1f}")
print("  output range :", f"{rebuilt.min():.1f} .. {rebuilt.max():.1f}")
print("  output shape :", rebuilt.shape)
assert rebuilt.shape == sample.shape
assert 0.0 <= rebuilt.min() and rebuilt.max() <= 255.0
assert rebuilt.max() > 1.5, "0..1 leaked out instead of pixels"

errors = ae.reconstruction_errors(model, sample)
print("  per-image MSE:", np.round(errors, 5))
assert errors.shape == (8,) and (errors >= 0).all()
worst = hardest_to_reconstruct(errors, 3)
print("  worst 3 idx  :", worst, "errors", np.round(errors[worst], 5))
assert errors[worst[0]] == errors.max()

codes = ae.latent_codes(result["encoder"], sample)
print("  latent codes :", codes.shape, "(expect (8, 16))")
assert codes.shape == (8, 16)

print("\n=== 6. noise helper matches the layer's units ===")
noisy = ae.add_noise(sample, 0.25, np.random.default_rng(0))
delta = float(np.abs(noisy - sample).mean())
print("  mean |change| :", f"{delta:.1f} pixel levels")
print("  clipped to    :", f"{noisy.min():.1f} .. {noisy.max():.1f}")
assert 0.0 <= noisy.min() and noisy.max() <= 255.0
assert delta > 5.0, "noise did nothing"

print("\n=== 7. anomaly holdout: a class the model never sees ===")
split = ae.split_anomaly(bundle, 2)
print(" ", split.describe())
assert split.anomaly_name == bundle.class_names[2]
assert len(split.normal_names) == 2
# nothing from the held-out class may appear in training
train_labels = np.argmax(bundle.y_train, axis=1)
assert len(split.x_train) == int((train_labels != 2).sum())

request = AutoencoderRequest(data=bundle, config=config, epochs=3,
                             batch_size=32, anomaly_class=2)
x_tr, x_va, sp = AutoencoderRun(request).arrays()
print("  train/val after holdout:", len(x_tr), "/", len(x_va),
      "| anomalies:", sp.n_anomalies)
assert sp is not None and len(x_tr) == len(split.x_train)

scores = ae.score_anomalies(model, split)
print("  normal mean  :", f"{scores['normal_mean']:.5f}")
print("  anomaly mean :", f"{scores['anomaly_mean']:.5f}")
print("  ratio / AUC  :", f"{scores['ratio']:.2f}x / {scores['auc']:.3f}")
assert 0.0 <= scores["auc"] <= 1.0
print("  --- readout ---")
print(ae.format_anomalies(scores))

print("\n=== 8. latent sweep, two arms, identical init ===")
sweep = LatentSweepRun(
    AutoencoderRequest(data=bundle, config=config, epochs=2, batch_size=32),
    latent_sizes=[2, 32],
    on_message=lambda m: None,
)
started = time.monotonic()
outcome = sweep.run()
print(f"  arms: {[a['label'] for a in outcome['arms']]} "
      f"in {time.monotonic()-started:.1f}s")
assert len(outcome["arms"]) == 2
assert [a["latent_dim"] for a in outcome["arms"]] == [2, 32]
print("  --- readout ---")
print(format_sweep(outcome))

print("\n=== 9. a sweep needs a real comparison ===")
try:
    LatentSweepRun(AutoencoderRequest(data=bundle, config=config), [8, 8])
    raise AssertionError("one distinct size should not be a sweep")
except ValueError as exc:
    print("  refused:", exc)

print("\n=== 10. core stays Qt-free ===")
dirty = [str(p) for p in (_bootstrap.ROOT / "nnstudio" / "core").rglob("*.py")
         if "PyQt6" in p.read_text(encoding="utf-8")]
print(" ", dirty or "clean")
assert not dirty

print("\nAUTOENCODER CORE OK")
