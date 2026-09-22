# AGENTS.md

Working brief for AI coding agents on this repository. Humans should read
`README.md` first; this file is about how to change the code without breaking
what it teaches.

---

## What this project is

An interactive desktop app for **learning** how neural networks work. Students
(15+) build a network with spin boxes and combos, train it on real data, and
watch what happens. PyQt6 for the interface, TensorFlow/Keras for the models.

The audience changes the engineering bar in one specific way: **this app must
not teach something false.** A number shown on screen, a claim in a log line or
a hint under a control is a teaching statement. Several of the invariants below
exist only because a plausible-looking shortcut would have quietly taught a
wrong lesson.

---

## Setup and commands

Everything runs inside `.venv`. TensorFlow pins exact versions of numpy,
protobuf and typing-extensions, so a global install eventually breaks an
unrelated project.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
pip install -r requirements.txt
```

| Task | Command |
|---|---|
| Run the app | `.\.venv\Scripts\python.exe main.py` |
| Syntax check everything | `.\.venv\Scripts\python.exe -c "import ast,pathlib;[ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('nnstudio').rglob('*.py')]"` |
| Import check | `.\.venv\Scripts\python.exe -c "import main"` |

There is no git repository here and no CI. Do not assume either.

---

## The one architectural rule

**`nnstudio/core/` contains no Qt.** Ever.

Everything in `core/` is plain Python that could run in a notebook, a script or
a test with no display. Everything in `ui/` is an adapter over it. This is what
makes the logic testable without a window, and it is the first thing to check
after any change:

```bash
.\.venv\Scripts\python.exe -c "import pathlib;print([str(p) for p in pathlib.Path('nnstudio/core').rglob('*.py') if 'PyQt6' in p.read_text(encoding='utf-8')] or 'clean')"
```

The two halves meet in exactly one place: `ui/workers.py`, which wraps a plain
`core` run object in a `QThread` and turns its callbacks into signals.

---

## Where new code goes

| You are adding… | It belongs in |
|---|---|
| A dataset source, encoding or split rule | `core/dataset.py` (tabular) or `core/vision.py` (images) |
| A layer type, activation or output head | `core/model_builder.py` (dense), `core/resnet.py` (conv) or `core/autoencoder.py` (AE) |
| A new kind of training run | `core/trainer.py`, `core/crossval.py`, `core/vision_trainer.py`, `core/autoencoder_trainer.py` |
| A control the user touches | the matching stage panel in `ui/`, `ui/vision/` or `ui/autoencoder/` |
| An image source shared by both image workspaces | `ui/image_source.py` |
| Orchestration between stages | `ui/dense_workspace.py`, `ui/vision_workspace.py` or `ui/autoencoder_workspace.py` |
| A background job | a `QThread` in `ui/workers.py` — never inline |
| A drawing | `ui/network_canvas.py`, `ui/resnet_canvas.py`, `ui/autoencoder_canvas.py`, `ui/image_grid.py`, `ui/pair_grid.py`, `ui/plots.py` |
| A colour or a widget style | `ui/theme.py` only. No inline stylesheets in panels. |

The three workspaces are deliberately independent. `DenseWorkspace`,
`VisionWorkspace` and `AutoencoderWorkspace` share the theme, the plotting
widgets and — between the two image workspaces — `ui/image_source.py`, and
nothing else. No shared model, no shared dataset, no shared worker, no
workspace importing from another workspace's panel folder. A convolutional
network is not a later stage of a dense one, and an autoencoder is not a later
stage of either. Keep it that way.

`ui/image_source.py` is the one piece of genuinely shared stage-1 machinery.
It lives at the top of `ui/` rather than inside `ui/vision/` precisely so that
the autoencoder workspace does not have to reach into the CNN's folder to use
it. If a third image workspace appears, it uses that panel too. If you ever
need a workspace-specific control in it, add the control to that workspace's
own architecture panel instead — `image_source.py` must stay ignorant of what
kind of network will consume the images.

---

## Invariants — do not break these

Each of these was either a bug that shipped, or a shortcut that would have
taught something false. There are fourteen.

### 1. Preprocessing is fit on training rows only

`_assemble()` in `core/dataset.py` fits the `ColumnTransformer` on
`train_idx` and only then transforms validation. Fitting on all rows first —
the obvious shortcut — leaks validation statistics (the scaler's mean, the
imputer's median) into training and inflates every score the app reports.

Guard: the training split's per-column mean is ~0 while the pooled mean is not.

```python
assert abs(bundle.x_train.mean(axis=0)).max() < 1e-5
assert abs(np.vstack([bundle.x_train, bundle.x_val]).mean(axis=0)).max() > 1e-3
```

### 2. Cross-validation leaves no model behind

K-fold trains k models and discards all of them. `_start_crossval()` in
`ui/dense_workspace.py` clears `self._model`, clears the history, and disables
both **Predict** and **Save model**. This shipped broken once: a model from an
earlier training run survived and the buttons stayed live, pointing at
something unrelated to the estimate on screen.

Cross-validation measures an architecture. It does not produce a model.

### 3. Every output head keeps its matching loss

`OUTPUT_MODES` in `core/model_builder.py` pairs each head with the loss that
belongs with it, and `MODES_BY_TASK` restricts which heads a detected task can
select. Softmax on a regression target is not a creative choice, it is a broken
model, and the UI must keep refusing it.

### 4. TensorFlow is imported lazily, inside the worker thread

Only `core/model_builder.keras_module()` imports Keras, and it is called from
the training thread. `main.py` sets `TF_CPP_MIN_LOG_LEVEL` before anything can
import it. A top-level `import tensorflow` anywhere in `ui/` or `main.py` adds
~8 seconds to startup — revert it.

### 5. Training never runs on the GUI thread

Always a `QThread` from `ui/workers.py`. `stop()` must genuinely stop: the
Keras callback checks the flag in `on_train_batch_end`, not only at epoch
boundaries, so Stop responds within a batch.

### 6. A frozen backbone is called with `training=False`

In `core/resnet.build_transfer()`. Without it the backbone's batch-norm
statistics keep drifting while it is supposedly frozen, and the head learns
against a moving target.

### 7. `use_skip=False` removes only the addition

`residual_block()` in `core/resnet.py` keeps identical convolutions, identical
depth and identical initialisation in both arms; only the `Add` disappears.
`SkipComparisonRun` re-seeds with `keras.utils.set_random_seed(1234)` before
each arm so the comparison is fair. If you touch the block, keep the ablation
honest — it is the one experiment the whole CNN workspace exists for.

The residual arm does carry ~3.7% more parameters (the 1x1 projections on
stride-2 stages). `format_comparison()` states this explicitly rather than
claiming the arms are identical. Keep that admission.

### 8. Combos go through `compact_combo()`

`QComboBox.minimumSizeHint()` is wide enough for its **longest item**. One
descriptive entry ("From scratch - build the residual stack yourself") forced a
666px minimum on a single combo, which pushed the sidebar's minimum past the
column it lives in — and Qt then compensated by squeezing every spin box below
its own minimum height, clipping the values inside.

That was a reported bug. A height symptom with a width cause. Any new combo
with descriptive entries goes through `compact_combo()` in `ui/widgets.py`.

### 9. Layout floors are load-bearing

- `min-height: 20px` on inputs in `ui/theme.py` stops any layout from crushing them.
- Tall panels wrap in `scrollable()` from `ui/widgets.py`, which uses
  `ScrollBarAsNeeded` horizontally — never `AlwaysOff`, which clips silently.
- Wrapping `hint()` labels use `QSizePolicy.Ignored` horizontally so they never
  dictate panel width.

### 10. Long builds report progress and can be cancelled

Anything that can take more than a couple of seconds - the CIFAR-10 download,
drawing thousands of shapes, reading a folder of photos - takes
`(on_progress, should_stop)` and honours both. A disabled button with no
feedback is indistinguishable from a crash; that was a real bug report.

`ensure_cifar10_archive()` in `core/vision.py` downloads into a `.part` file
and only renames it once the SHA-256 matches, so an interrupted transfer can
never be mistaken for a finished one. It resumes with a `Range` header, which
matters because the host serves at ~0.1 MB/s and a full transfer is ~30
minutes. Do not replace it with `keras.datasets.cifar10.load_data()` alone:
that has no progress, no cancel and no resume.

### 11. Preparation settings are reused, not re-read

`DataPanel.current_spec()` returns the `PreparationSpec` the loaded bundle was
actually built with. Cross-validation reuses it so the folds match the
holdout's settings instead of whatever the widgets happen to show now.

---

### 12. The autoencoder's target is its own input — never a label

`AutoencoderRun.run()` calls `model.fit(x_train, targets(x_train), ...)` where
`targets()` is just `images / 255.0`. Class names reach the autoencoder
workspace for exactly one purpose: choosing which images `split_anomaly()`
withholds. If a label ever enters the loss, the workspace stops demonstrating
unsupervised learning while still claiming to.

Guard: the training log line "No labels are used anywhere in this run" is
asserted by the GUI check, and `y_train` must not appear in
`core/autoencoder_trainer.py` at all.

### 13. The autoencoder maps 0..255 in to 0..1 out, and only `reconstruct()` undoes it

`ImageBundle` holds pixels as float32 in 0..255, so the model rescales on the
way in. The decoder ends on a **sigmoid**, so the output is 0..1 — which keeps
the MSE legible (`0.0117`, not `760`). That asymmetry is deliberate, and the
multiplication back to displayable pixels lives in exactly one place,
`autoencoder.reconstruct()`. Do not scatter `* 255` through the UI, and do not
"fix" the asymmetry by adding a `Rescaling(255)` output layer — that makes
every loss on screen unreadable and breaks the give-up comparison below.

For a denoising run the model is fed the corrupted image and must be scored
against the **clean** one, which is why `reconstruction_errors()` takes an
`against=` argument. Scoring it against its own noisy input would reward the
network for faithfully reproducing the damage.

### 14. No readout claims a lesson the run did not earn

`mean_image_baseline()` computes the MSE of answering every image with the
average of the training set. That is the score for giving up, and it is shown
next to every autoencoder loss — on the metric row, on the learning-curve
chart as a dotted line, and in the log.

`format_sweep()` uses it as a gate. While no arm has beaten the floor it
reports that **nothing can be concluded**, rather than reporting a tie:

```
latent  2  (1536x squeeze):  val MSE 0.14816
latent 32  (  96x squeeze):  val MSE 0.14783
   giving up                 val MSE 0.05440
NO CONCLUSION IS AVAILABLE FROM THIS RUN.
```

Those two arms differ by 0.2%, which a naive formatter would print as "the
bottleneck made no difference" — a false lesson, since neither arm had started
learning. Arms that finish within `MEANINGFUL_GAIN` (15%) of each other are
reported as *not separated*, never as a win. This invariant exists because the
first version of this formatter did claim a win from a 2-epoch sweep.

---

## How to verify a change

Start with the committed suites — `python tests/run_all.py`, about two and a
half minutes on CPU, or `--fast` for the three that need no display. They are
standalone scripts, not pytest; `tests/README.md` says why. Add to them when you
add behaviour.

For anything they do not cover yet, verify headlessly like this.

### Core logic — no display needed

```python
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from nnstudio.core import dataset as ds
from nnstudio.core.model_builder import LayerSpec, NetworkConfig
from nnstudio.core.trainer import TrainingRequest, TrainingRun

frame, target = ds.load_builtin("Iris - 4 features, 3 classes")
bundle = ds.PreparationSpec(frame, target, name="Iris").prepare()
cfg = NetworkConfig(bundle.n_inputs, bundle.n_outputs,
                    [LayerSpec(16, "relu")], "softmax", "adam", 0.01)
print(TrainingRun(TrainingRequest(cfg, bundle, epochs=3)).run()["final_scores"])
```

### UI — offscreen Qt

```python
import os; os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication
from nnstudio.ui.main_window import MainWindow
from nnstudio.ui.theme import apply_theme
app = QApplication([]); apply_theme(app)
win = MainWindow(); win.show(); app.processEvents()
```

Drive real widgets (`panel.build_button.click()`), not private methods, so the
signal wiring is exercised. For a run that finishes on a thread, block on a
`QEventLoop` connected to `worker.finished`, with a `QTimer.singleShot`
timeout as a safety net.

Render a frame with `QPixmap(win.size()); win.render(pix); pix.save(path)`.

### A shell caveat that has cost time three times

Never write `
` inside a Python patch script fed through a bash heredoc. The
escape collapses to a REAL newline and silently breaks the string literal it
was inside, producing `SyntaxError: unterminated f-string literal` several
edits later. Build the backslash instead:

```python
NL = chr(92) + "n"
new = '    return "' + NL + '".join(lines)'
```

Better still, use the Write or Edit tools for anything containing escapes.
This has bitten three separate edits in this repository.

### Two measurement caveats

- **Offscreen has no Segoe UI.** It falls back to a font with much wider
  metrics, so `minimumSizeHint().width()` readings are inflated roughly 2x.
  **Height** readings offscreen do match a real screen and are trustworthy.
  Never chase a width number from an offscreen render — confirm with the user.
- Modal dialogs block a headless run. Patch them in the module under test:
  `module.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)`.

---

## Conventions

- **English** for code, identifiers, comments, UI copy, logs and docs.
  `manual.html` is the single deliberate exception: it is Spanish because its
  audience is the instructor's Spanish-speaking students, and it keeps the UI
  control names in English because that is what is on the screen.
- Comments explain **why**, not what. A comment restating the line below it is
  noise; a comment explaining why the shortcut was rejected is the valuable one.
- No emoji anywhere in code, UI or logs.
- Docstrings on every module and on any function whose purpose is not obvious
  from its name.
- Dataclasses for configuration objects; they cross the core/ui boundary and
  `dataclasses.replace()` is used to vary one field.
- Errors surface to the user through a `QMessageBox` **and** the log, with the
  first paragraph of the message as the dialog text.

---

## Measured facts — do not contradict these

These numbers appear in the README, the app's own log output and `manual.html`.
They were measured in this project, not copied from a paper. If you change
defaults or the model code, re-measure before editing the claims.

| Claim | Value |
|---|---|
| Skip connections, 8 layers, 8 epochs | residual +29 points |
| Skip connections, 20 layers, 30 epochs | **tied at 0.9944** — plain caught up |
| Skip connections, 50 layers, 18 epochs | residual val 0.978 / train 1.000; plain val 0.639 / **train 0.943** |
| Transfer learning, 240 images, 3 epochs | 100% val accuracy, 131,331 of 23,719,043 trainable (**0.55%**) |
| Wine overfitting | `val_loss` minimum at epoch 60, rising to 0.243 by epoch 300 |
| Wine validation split | 36 rows, so one sample is worth **2.78 accuracy points** |
| Autoencoder sweep, 900 shapes at 32px, 25 epochs/arm | latent 2 → 0.0417, latent 8 → 0.0167, latent 64 → 0.0117 val MSE |
| The give-up floor for that dataset | **0.0561** — so latent 2 is only 1.3x better than not trying |
| When the sweep arms separate | still identical at epoch 2; 50% apart by epoch **7**; clearly ordered by 15 |
| Autoencoder cost on CPU | ~0.7s per epoch at 720 images; all three sweep arms in **61s** |
| Colour recovery, latent 64 | red-minus-blue correlation +0.03 at 25 epochs, **+0.91 at 90** |
| Colour recovery, latent 8 | +0.06 even at 90 epochs — that waist never affords colour |
| When each waist beats the give-up floor | latent 64 at epoch **9**, latent 8 at **13**, latent 2 at **19** |
| Anomaly detection, hold out `triangle`, 25 epochs, latent 8 | **ROC AUC 0.718 +/- 0.076** over 6 seeds (0.604 … 0.836) |
| The same, latent 64 | **ROC AUC 0.453 +/- 0.104** over 6 seeds (0.336 … 0.611) — below a coin flip on 4 of 6 |
| Latent 8 beats latent 64 as a detector | **6 of 6 seeds**, mean AUC gap +0.265 +/- 0.099 |
| Representative single run (seed 7, the median) | latent 8: familiar 0.01864, held-out 0.02254, 1.21x, AUC 0.726 |

The 50-layer row is the important one, and the diagnostic is **training**
accuracy: a deep plain stack that cannot fit its own training data is failing
to optimise, not overfitting. Below ~20 layers the shortcut mostly buys
convergence speed — `format_comparison()` detects which of the two lessons
applies and says so. Do not simplify that into "plain networks cannot learn".

The two colour rows are the autoencoder's equivalent trap. The reconstructions
come back grey, and there are **two different causes that look identical on
screen**: at latent 64 colour is merely unconverged and arrives by ~90 epochs,
while at latent 8 it never arrives at all. `COLOUR_NOTE` in
`core/autoencoder_trainer.py` keeps them apart. Do not collapse it into "the
bottleneck loses colour" — that would be wrong in one of the two cases, and it
is the case a curious student is most likely to test.

The anomaly rows carry the workspace's most counter-intuitive result, and it
survives a seed sweep: **the autoencoder that reconstructs better is the worse
detector.** Latent 8 reconstructs 1.28x *worse* than latent 64 and detects
+0.265 of AUC better, and it wins on **6 of 6 seeds** — a gap 2.7x its own
spread. A wide waist generalises well enough to rebuild the class it never
saw; on 4 of those 6 seeds latent 64 landed *below* 0.5, meaning the unseen
triangles rebuilt BETTER than the familiar classes and the error ranked them
backwards. Anomaly detection does not want the best reconstructor, it wants
one tight enough that only the familiar comes out right.
`format_anomalies()` already points a weak separation at the latent size; do
not "improve" it by suggesting a wider waist.

**How this section nearly shipped a flattered number.** The first version of
these rows recorded AUC 0.836 for latent 8 and 0.611 for latent 64, each from
a single seeded run. Both were the *best of six*. The conclusion held, but the
headline figures did not, and the app's own manual has an experiment about
exactly this failure ("cambiá el seed y perdé la fe"). Any AUC, accuracy or
gap that reaches these docs from one run is provisional until a seed sweep
agrees with it. `format_anomalies()` now prints the spread so a student
reading one run sees it too.

---

## Known gaps

Honest list, roughly by value:

1. **Test coverage is autoencoder-heavy.** `tests/` exists now (six suites, see
   `tests/README.md`), but four of the six are about the autoencoder. The dense
   and convolutional workspaces get one shared regression suite that only proves
   they still train. Nothing covers k-fold cross-validation, the preprocessing
   leak it was written to fix, dataset loading from CSV/Excel, or the ResNet50
   transfer path — that last one needs a 98 MB download, so it may be better as
   an opt-in suite than a default one.
2. **No decision-boundary plot** for the 2D datasets (two moons, circles).
   It would make the "you need hidden layers" lesson visual instead of numeric.
3. **No gradient-boosting baseline** next to the dense network. Measurements
   during development showed trees beating the MLP on most tabular shapes; a
   built-in baseline would make that comparison available to students.
4. **No feature-map visualisation** for the CNN. Showing what the first
   convolution actually responds to is the most "see how it works" thing still
   missing.
5. `nnstudio/ui/vision_workspace.py`, `dense_workspace.py` and
   `autoencoder_workspace.py` share a fair
   amount of orchestration shape. Extracting a common base is tempting —
   resist it unless the duplication actually hurts, because the independence of
   the three paths is the point.
