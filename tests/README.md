# Tests

Ten suites. Run them all:

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
| `test_sequence_core.py` | ~27s | The four sequence tasks, their floors, and the order probe |
| `test_transformer_core.py` | ~84s | The token tasks, the position probe, and all four map verdicts |
| `test_workspaces_regression.py` | ~15s | Dense and convolutional still train after shared-code changes |
| `test_judge_gui.py` | ~25s | The Choose/Judge buttons, end to end |
| `test_autoencoder_gui.py` | ~38s | The whole autoencoder workspace, through its real widgets |
| `test_sequence_gui.py` | ~46s | The whole sequence workspace, including both probe verdicts |
| `test_transformer_gui.py` | ~132s | The whole transformer workspace, down to switching probe arms |

Between four and a half and seven and a half minutes for the lot on CPU,
depending on what else the machine is doing (265s and 437s measured).

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

The sequence suites follow the same rule. They assert the *verdict* the probe
reaches (`ORDER MATTERS HERE` on the spike task, `ORDER CARRIED NOTHING` on the
counting one) and that a random walk is never beaten, because those survive a
reseed. They never assert a particular accuracy.

The transformer suites assert the map's *verdict* — faithful, decorative,
distributed, moot — each on a model that genuinely earns it, and that the
most-attended column lands on the answer for most sequences rather than for
any particular one.

## Conventions

- `_bootstrap.py` is the only shared module: import path, quiet TensorFlow, and
  `offscreen()`, which must run **before** PyQt6 is imported.
- `fixtures.py` draws the hand-made images instead of committing binaries, so a
  fresh clone can run everything with nothing but the repository.
- Generated files — screenshots and those drawn images — go to `tests/_output/`,
  which is gitignored.
- Offscreen Qt ships with no fonts, and without them **width** measurements
  come out about 1.8x the real ones. `offscreen()` points `QT_QPA_FONTDIR` at the
  Windows fonts, which fixes that and makes the screenshots readable. Heights
  were already right either way.
- A widget on a tab that is not in front reports `isVisible() == False`, which
  makes any `assert not w.isVisible()` pass blind. Use `isVisibleTo(panel)`.
