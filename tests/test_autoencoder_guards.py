"""The readout must refuse to claim a lesson it did not measure.

Invariant 14 in AGENTS.md, defended here. A run that learned nothing must say so
instead of dressing up noise as a finding, and the give-up floor is what tells
the two apart. Sections 4 through 6 feed the formatter hand-built numbers - the
real measured ones, a tie, and an inversion - so the wording is checked without
paying for another forty epochs of training.
"""
import _bootstrap  # noqa: F401 - import path and quiet TensorFlow

from nnstudio.core import vision as vz
from nnstudio.core import autoencoder as ae
from nnstudio.core.autoencoder_trainer import (
    AutoencoderRequest, AutoencoderRun, LatentSweepRun,
    format_run, format_sweep, SWEEP_EPOCHS, SWEEP_SIZES,
)

bundle = vz.make_synthetic(vz.SyntheticSpec(n_images=240, image_size=32,
                                            n_classes=3, noise=0.05, seed=3))
config = ae.AutoencoderConfig(input_shape=bundle.input_shape, latent_dim=16,
                              stages=2, base_filters=8)

print("=== 1. the mean-image baseline is a real floor ===")
base = ae.mean_image_baseline(bundle.x_train, bundle.x_val)
print("  baseline MSE:", f"{base:.5f}")
assert 0.05 < base < 0.30, base
# An untrained model must land at or above that floor, never below it.
untrained, _ = ae.build_autoencoder(config)
raw = ae.reconstruction_errors(untrained, bundle.x_val).mean()
print("  untrained model:", f"{raw:.5f}", "(must not beat the floor by much)")
assert raw > base * 0.5

print("\n=== 2. a 2-epoch sweep must REFUSE to conclude ===")
sweep = LatentSweepRun(
    AutoencoderRequest(data=bundle, config=config, epochs=2, batch_size=32),
    latent_sizes=[2, 32], on_message=lambda m: None,
)
short = sweep.run()
text = format_sweep(short)
print(text)
assert "NO CONCLUSION IS AVAILABLE" in text
assert "cut the error" not in text, "it claimed a win from a 2-epoch run"
print("\n  -> refused correctly")

print("\n=== 3. a single 2-epoch run must say it learned nothing ===")
run = AutoencoderRun(
    AutoencoderRequest(data=bundle, config=config, epochs=2, batch_size=32),
    on_message=lambda m: None,
)
result = run.run()
text = format_run(result)
print(text)
assert ("LEARNED ESSENTIALLY NOTHING" in text
        or "WORSE THAN GIVING UP" in text), text

print("\n=== 4. the measured 40-epoch numbers DO produce the lesson ===")
# Real values measured on 900 images, 32px, conv, 2 stages, 16 filters.
measured = {
    "n_inputs": 3072,
    "arms": [
        {"latent_dim": 2, "compression": 1536.0, "trainable_params": 29733,
         "epochs_run": 40, "baseline": 0.14405,
         "final_scores": {"loss": 0.03256}, "history": {}},
        {"latent_dim": 8, "compression": 384.0, "trainable_params": 54315,
         "epochs_run": 40, "baseline": 0.14405,
         "final_scores": {"loss": 0.01340}, "history": {}},
        {"latent_dim": 64, "compression": 48.0, "trainable_params": 283747,
         "epochs_run": 40, "baseline": 0.14405,
         "final_scores": {"loss": 0.00848}, "history": {}},
    ],
}
text = format_sweep(measured)
print(text)
assert "cut the error 3.8x" in text, text
assert "COLOUR does not" in text, "the measured colour note is missing"
assert "NO CONCLUSION" not in text
assert "did not fall cleanly" not in text, "these numbers ARE monotonic"

print("\n=== 5. a genuine tie must be reported as a tie, not a win ===")
tie = {
    "n_inputs": 3072,
    "arms": [
        {"latent_dim": 8, "compression": 384.0, "trainable_params": 1000,
         "epochs_run": 30, "baseline": 0.14405,
         "final_scores": {"loss": 0.0200}, "history": {}},
        {"latent_dim": 64, "compression": 48.0, "trainable_params": 2000,
         "epochs_run": 30, "baseline": 0.14405,
         "final_scores": {"loss": 0.0195}, "history": {}},
    ],
}
text = format_sweep(tie)
print(text)
assert "did NOT separate" in text
assert "cut the error" not in text

print("\n=== 6. a wider waist doing WORSE must be said plainly ===")
worse = {
    "n_inputs": 3072,
    "arms": [
        {"latent_dim": 8, "compression": 384.0, "trainable_params": 1000,
         "epochs_run": 30, "baseline": 0.14405,
         "final_scores": {"loss": 0.0200}, "history": {}},
        {"latent_dim": 64, "compression": 48.0, "trainable_params": 2000,
         "epochs_run": 30, "baseline": 0.14405,
         "final_scores": {"loss": 0.0400}, "history": {}},
    ],
}
text = format_sweep(worse)
print(text)
assert "did worse than" in text

print("\n=== 7. the preset points at a measured number ===")
print("  SWEEP_EPOCHS:", SWEEP_EPOCHS, "| SWEEP_SIZES:", SWEEP_SIZES)
assert SWEEP_EPOCHS >= 15, "below 15 epochs the arms had not separated"

print("\nGUARDS OK")
