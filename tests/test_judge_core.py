"""Load an image from disk and ask the model whether it is an anomaly.

No Qt here: this is load_single_image + judge_image + format_judgement, driven
the way the workspace drives them. The model is trained with 'triangle' held
out, so the triangle is the one case whose verdict carries information.
"""
import _bootstrap

import fixtures
from nnstudio.core import vision as vz
from nnstudio.core import autoencoder as ae
from nnstudio.core.autoencoder_trainer import AutoencoderRequest, AutoencoderRun

HERE = _bootstrap.output_dir()

# The model sees circles and squares. Never a triangle.
bundle = vz.make_synthetic(vz.SyntheticSpec(n_images=900, image_size=32,
                                            n_classes=3, noise=0.05, seed=7))
cfg = ae.AutoencoderConfig(input_shape=bundle.input_shape, latent_dim=8,
                           stages=2, base_filters=16)
req = AutoencoderRequest(data=bundle, config=cfg, epochs=25, batch_size=32,
                         anomaly_class=2)          # hide 'triangle'
run = AutoencoderRun(req, on_message=lambda m: None)
result = run.run()
x_train, x_val, split = run.arrays()
print(split.describe())
print(f"val MSE on the familiar classes: {result['final_scores']['loss']:.5f}\n")
model, encoder = result["model"], result["encoder"]

cases = fixtures.make_all(HERE)

print("=== 1. load_single_image gives it the dataset's shape ===")
img = vz.load_single_image(str(cases[0][0]), image_size=32, channels=3)
print("   a 256x256 file ->", img.shape, "| range", f"{img.min():.0f}..{img.max():.0f}")
assert img.shape == (1, 32, 32, 3)

print("\n=== 2. the verdict, case by case ===")
print(f"   {'image':<18} {'error':>9} {'percentile':>11}  {'sigmas':>7}   what it is")
for path, what in cases:
    img = vz.load_single_image(str(path), 32, 3)
    v = ae.judge_image(model, img, x_val)
    print(f"   {path.name:<18} {v['error']:>9.5f} {v['percentile']:>10.0f}% "
          f"{v['sigmas']:>+7.1f}   {what}")

# Stripes and solid white share nothing with the dataset at any seed. The
# triangle does not get an assertion here on purpose: across four seeds it
# landed anywhere from the 63rd to the 91st percentile, and a test that pinned
# it would be pinning one seed's luck.
for name in ("my_stripes.png", "my_white.png"):
    img = vz.load_single_image(str(HERE / name), 32, 3)
    assert ae.judge_image(model, img, x_val)["percentile"] >= 95, name

print("\n=== 3. the full readout for the triangle ===")
img = vz.load_single_image(str(HERE / "my_triangle.png"), 32, 3)
v = ae.judge_image(model, img, x_val)
code = ae.latent_codes(encoder, img)
print(ae.format_judgement(v, code))
assert "latent code" in ae.format_judgement(v, code)

print("\n=== 4. and for one it DOES know ===")
img = vz.load_single_image(str(HERE / "my_circle.png"), 32, 3)
print(ae.format_judgement(ae.judge_image(model, img, x_val)))

print("\n=== 5. errors are handled ===")
for path, why in ((HERE / "does_not_exist.png", "missing file"),
                  (__file__, "not an image")):
    try:
        vz.load_single_image(str(path), 32, 3)
        raise AssertionError(f"{why}: should have raised")
    except vz.VisionError as exc:
        print(f"   {why:<14} -> {str(exc)[:72]}")
try:
    ae.judge_image(model, img, x_val[:3])
    raise AssertionError("3 reference images should not be enough")
except ae.AutoencoderError as exc:
    print(f"   {'tiny reference':<14} -> {exc}")

print("\nJUDGE CORE OK")
