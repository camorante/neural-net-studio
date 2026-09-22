# Tests

Six suites. Run them all:

```bash
python tests/run_all.py
```

Or only the ones that need no display:

```bash
python tests/run_all.py --fast
```

Or one by name: `python tests/run_all.py judge_gui`, or just
`python tests/test_judge_gui.py` — every suite is a standalone script.

| Suite | Time | What it defends |
|---|---|---|
| `test_autoencoder_guards.py` | ~30s | The readout refuses to claim a lesson it did not measure |
| `test_autoencoder_core.py` | ~17s | Validation, geometry, both variants, a real run, anomalies, the sweep |
| `test_judge_core.py` | ~22s | `load_single_image` + `judge_image` + `format_judgement` |
| `test_workspaces_regression.py` | ~15s | Dense and convolutional still train after shared-code changes |
| `test_judge_gui.py` | ~25s | The Choose/Judge buttons, end to end |
| `test_autoencoder_gui.py` | ~38s | The whole autoencoder workspace, through its real widgets |

About two and a half minutes for the lot, on CPU.

## Why these are scripts and not pytest

They train real networks and drive a real Qt application. A suite here is a
transcript: you run it and the printed numbers are the evidence, with the
assertions sitting next to the value they are about. Collapsing that into
`assert result == expected` would keep the check and throw away the reason
anyone would trust it.

They are also slow by nature. There is no unit-test fiction available: an
autoencoder that has trained for two epochs genuinely has not learned anything,
and the suite that proves the app *says so* has to pay for those two epochs.

## What the suites will not assert

Single-run metrics are anecdotes. Measured across six seeds, ROC AUC at latent 8
ranged 0.604 to 0.836, and the same hand-drawn circle landed anywhere between
the 28th and the 78th percentile across four seeds. So the suites assert what
survives a reseed — stripes and solid white above the 95th percentile, latent 8
detecting better than latent 64 — and never a third decimal.

## Conventions

- `_bootstrap.py` is the only shared module: import path, quiet TensorFlow, and
  `offscreen()`, which must run **before** PyQt6 is imported.
- `fixtures.py` draws the hand-made images instead of committing binaries, so a
  fresh clone can run everything with nothing but the repository.
- Generated files — screenshots and those drawn images — go to `tests/_output/`,
  which is gitignored.
- Offscreen Qt has no Segoe UI, so **width** measurements come out roughly twice
  the real ones. Only **height** is trustworthy there.
