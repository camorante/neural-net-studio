# Neural Net Studio

An interactive desktop playground for building, training and inspecting neural
networks — both **fully connected** ones over tabular data and **convolutional**
ones over images. Design the architecture with spin boxes and combos, watch the
diagram redraw as you type, then train on real data and see the learning curves
move epoch by epoch.

Built with **PyQt6** for the interface and **TensorFlow / Keras** for the models.
No code to write: you move numbers and watch what changes.

It is built for teaching, which sets one unusual requirement — **the app must
not teach something false.** Several design decisions here exist only because
the convenient shortcut would have quietly taught a wrong lesson. Those are
called out where they appear.

### Where to start

| You want to… | Go to |
|---|---|
| Learn the concepts, in Spanish, with diagrams | [`manual.html`](manual.html) |
| Install and run it | [Install](#install) below |
| Understand the code before changing it | [`AGENTS.md`](AGENTS.md) |
| Just try something and break it | [Experiments](#experiments-worth-running) |

## Requirements

Python **3.10 – 3.12**, and roughly 2 GB of disk for the dependencies.

| Package | Used for |
|---|---|
| `PyQt6` | the whole interface |
| `tensorflow` | building and training the models |
| `scikit-learn` | datasets, encoding, splits, k-fold |
| `pandas` | reading and holding tabular data |
| `numpy` | arrays everywhere |
| `matplotlib` | learning curves and fold charts |
| `openpyxl` | reading `.xlsx` files |
| `pillow` | drawing the synthetic shapes (arrives with matplotlib) |

TensorFlow on native Windows is CPU-only. That is fine here — everything in
this app is sized so a CPU can finish it while you watch.

## Install

Python 3.10-3.12. Always use a virtual environment: TensorFlow pins exact
versions of numpy, protobuf and typing-extensions, and installing it globally
will eventually break an unrelated project.

Windows (PowerShell):

```bash
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

macOS / Linux / Git Bash:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

TensorFlow on native Windows runs on CPU only, which is more than enough for
the network sizes this app is meant for.

## Run

With the environment activated:

```bash
python main.py
```

Or without activating it, straight from the interpreter in the venv:

```bash
.\.venv\Scripts\python.exe main.py
```

The window opens immediately, but the first Train click takes a few seconds
before epoch 1 - that is TensorFlow loading. The import is deferred into the
training thread on purpose, so startup never blocks on it. The log says
`Loading TensorFlow...` while it happens.

## Two workspaces

The window opens on a mode selector, not a step list:

```
[ Dense network (NN) ]   [ Convolutional (CNN) ]
        |                          |
   1. Data                    1. Images
   2. Architecture            2. Architecture
   3. Training                3. Training
   4. Predict                 4. Predict
```

A convolutional network is not a later stage of a dense one - it is a
different path, with different data, a different architecture vocabulary and
different failure modes. So each gets its own four stages, its own model, its
own workers and its own diagrams. Training one leaves the other untouched.

## The dense workspace

**1. Data** - Start from a built-in dataset (Iris, Wine, Breast cancer, Digits,
Diabetes, two moons, concentric circles, a noisy sine) or load your own CSV,
TSV, Excel, JSON or Parquet file. Pick the target column; the app guesses
whether it is a binary, multiclass or regression problem and you can override
it. Adaptation handles the rest:

- missing numeric values filled with the column median, categoricals with the mode
- numeric features standardised (mean 0, std 1)
- categorical features one-hot encoded, so `city` becomes `city_ba`, `city_cba`, ...
- columns with more than 50 distinct categories are dropped and reported
- a stratified train/validation split

The encoded feature count becomes your input layer size. That is why the input
spin box locks once data is prepared - the network cannot have a different
number of inputs than the data has columns.

**2. Architecture** - Add and remove hidden layers freely (up to 12), each with
its own neuron count, activation, dropout rate and optional batch
normalisation. The parameter counter updates live, and warnings appear when the
configuration does not make sense.

The output layer offers five heads, each paired with the loss that belongs
with it:

| Head | Meaning | Loss |
|---|---|---|
| Softmax | one neuron per class, outputs sum to 1 | categorical crossentropy |
| Sigmoid | one neuron squashed into 0..1, above 0.5 is positive | binary crossentropy |
| Linear | any real number | mean squared error |
| ReLU | clamps negatives to 0, for targets that cannot be negative | mean squared error |
| Softplus | smooth ReLU, always positive and always differentiable | mean squared error |

Only the heads that fit the detected problem stay selectable. Choosing softmax
for a regression target is not a creative choice, it is a broken model.

**3. Training** - Set epochs, batch size, shuffling and optional early stopping
with patience. Training runs on a background thread, so the window stays
responsive and Stop actually stops. The diagram pulses along its connections
while the model learns; loss and metric curves redraw as epochs land. Save the
trained model as a `.keras` file when you are happy with it.

### Cross-validation

A single holdout split gives you one number from one random draw. On a small
dataset that number is mostly luck: with 36 validation rows, one sample is
worth 2.8 accuracy points, so 97.2% and 100% differ by a single row.

K-fold answers that. Set `k` in the Training tab and press **Cross-validate**:
the data is cut into k parts, k models are trained, and each part serves as
validation exactly once. Every row is validated, none twice.

What you get back is `mean +/- std` and a bar per fold. **The spread is the
point.** A tall band means the single number you were reading was noise, and
that two architectures three points apart are indistinguishable.

Three things worth being clear about:

- **The k models are thrown away.** Cross-validation measures an architecture,
  it does not produce a model. Once you trust the estimate, press Train once to
  get something you can actually predict with. The app enforces this: Predict
  and Save model stay disabled after a cross-validation.
- **It costs k times a normal run.** The app tells you before starting.
- **Stratified by default.** Class proportions are kept in every fold. If some
  class has fewer than k members, it falls back to plain K-fold and says so in
  the log - a fold with zero examples of a class is not a measurement.

### A note on leakage

Preprocessing is fit on the training rows of each split only, then applied to
validation. Fitting the scaler and the imputer on all rows first - the obvious
shortcut - leaks validation statistics into training and quietly inflates every
score the app reports. Expect slightly lower, more honest numbers here than
from a pipeline that skips this.

**4. Predict** - Type raw feature values, or pull a random row out of the
validation split and compare the prediction against the true value. Your input
goes through exactly the same imputation, scaling and encoding the model was
trained on.

## Experiments worth running

Breaking something on purpose teaches more than watching it work. Each of these
takes a couple of minutes.

| # | Do this | What you learn |
|---|---|---|
| 1 | **Clear all** hidden layers, train on `Two moons` | What is left is logistic regression: it can only draw a straight line, and two interlocking moons do not separate with one. |
| 2 | Set `Learning rate` to `0.5` | The loss diverges instead of descending — the step is so big it overshoots the minimum every time. Then try `0.00001` and watch it learn nothing. |
| 3 | `Positive sine`, tick **Standardise the target**, pick a **ReLU** output | The loss stalls. Standardising made half the targets negative and ReLU clamps those to zero, so the model *cannot* reach them. Switch to **Softplus** and it works. |
| 4 | `Wine`, 300 epochs, early stopping off | `val_loss` bottoms out near epoch 60 and climbs back to 0.243 while training loss goes to zero. That is overfitting, live. |
| 5 | `Wine`, change `Random seed` to 0, 1, 7, 99 | The accuracy moves several points with nothing else changed. With 36 validation rows one sample is worth 2.78 points. Then run **Cross-validate**. |
| 6 | CNN: **Compare skip on/off**, then **Preset: 50 layers** and compare again | At 8 layers the shortcut buys speed. At 50 the plain arm cannot fit even its training data — that is the failure ResNet was built to fix. |
| 7 | CNN: turn **Move** off and rebuild the dataset | Without translation there is nothing for translation invariance to buy. An architecture is not better in the abstract, only for a shape of data. |

`manual.html` walks through all seven in Spanish, with what to expect and why.

## The convolutional workspace

Four stages of its own, mirroring the dense side.

### 1. Images

Three sources. **Synthetic shapes** draws circles, squares, triangles and
crosses at random positions, sizes and angles, entirely offline - random
placement is what makes translation invariance worth having, and therefore
what a convolution buys you over a dense layer. **CIFAR-10** pulls real
photographs (asks first: ~170 MB). **Image folder** reads a local folder with
one subfolder per class.

### 2. Architecture - from scratch

The residual block is written out by hand rather than imported, and
`Use skip connections` toggles only the addition - same convolutions, same
depth, same initial weights. The diagram draws the shortcut as an arc that
bypasses both convolutions, and it disappears when you switch it off.

`Compare skip on/off` trains both arms back to back from identical
initialisation and overlays the curves.

**What the comparison actually shows, measured on this machine** (900 synthetic
shapes, 32x32, 3 classes):

| Depth | Result |
|---|---|
| 8 layers | residual +29 points after 8 epochs |
| 20 layers | residual +17 after 8 epochs, but **tied at 0.9944 after 30** - the plain arm caught up and even converged sooner |
| 50 layers | residual val 0.978 / **train 1.000**; plain val 0.639 / **train 0.943** |

Read the training accuracy in that last row. At 50 layers the plain stack
cannot fit its own training data. That is an optimisation failure, not
overfitting - more depth made the network harder to *train*. That is precisely
the degradation ResNet was invented to fix, and why the answer was a shortcut
rather than more regularisation.

Below roughly 20 layers the shortcut mostly buys convergence speed, because
BatchNorm already fixed much of what made moderately deep plain networks fail.
Do not let a shallow run convince you that plain networks cannot learn. Use the
**Preset: 50 layers** button to reach the depth where the real effect appears
(about 2 minutes per arm).

### 2. Architecture - transfer learning

A pretrained ResNet50 or ResNet50V2 backbone with a small trainable head. The
first run downloads ~98 MB of ImageNet weights into `~/.keras/models` and asks
before doing it.

Measured here: 240 training images, 3 epochs, **100% validation accuracy**,
with 131,331 trainable parameters out of 23,719,043 - **0.55% of the network
learns**. The other 99.45% arrived already knowing edges, textures and shapes.
That is the honest reason "neural networks need big datasets" is only half
true: the big dataset existed, it just was not yours.

A frozen backbone is also kept in inference mode, so its batch-norm statistics
do not drift while it is supposedly frozen.

## Troubleshooting

**The first Train click hangs for several seconds.**
That is TensorFlow loading. The import is deferred into the training thread on
purpose so the window opens instantly instead of blocking on startup. The log
prints `Loading TensorFlow...` while it happens. It only costs you once.

**`Activate.ps1` is blocked by the execution policy.**
Skip activation entirely and call the interpreter inside the environment:
`.\.venv\Scripts\python.exe main.py`.

**A warning about GPU support on Windows.**
Expected. TensorFlow ≥ 2.11 is CPU-only on native Windows. Nothing here needs
a GPU.

**Training is slower than you expected in the CNN workspace.**
Convolutions are far heavier than dense layers. Drop the image count, the image
size or `Blocks/stage` — in that order. The 50-layer preset takes about two
minutes per arm with 900 images at 32px, which is the intended budget.

**`Compare skip on/off` is greyed out.**
It only applies to a from-scratch stack. Switch the model mode back from
transfer learning.

**Predict and Save model went dark after a cross-validation.**
Deliberate. K-fold trains k models and throws all of them away; what it
produces is an estimate, not a model. Train once normally to get one.

**The app says it wants to download something.**
Only two things ever download, both on first use and both after asking:
CIFAR-10 (~170 MB, into `~/.keras/datasets`) and the ImageNet weights for
ResNet50 (~98 MB, into `~/.keras/models`). The synthetic shapes need neither.

**The CIFAR-10 download takes forever.**
It does, and that is the server, not the app. The university host that serves
CIFAR-10 runs at roughly **0.1 MB/s**, so 170 MB takes about **30 minutes**.
The dialog says so before you start.

What the app gives you while it runs: a live progress bar with megabytes
downloaded, current speed and an estimated time left, plus a **Cancel** button
that works immediately. Cancelling keeps what was already downloaded, and
pressing **Build dataset** again resumes from that point instead of starting
over - the server supports range requests, so nothing is wasted.

The archive is only considered valid once its SHA-256 matches, so a partial
transfer can never be mistaken for a finished one. Start with **Synthetic
shapes** while it downloads; they need no network at all.

## Layout

```
main.py                        entry point, quiets TensorFlow before any import
manual.html                    illustrated manual for students (Spanish)
AGENTS.md                      brief for AI agents working on this code
requirements.txt               pinned minimums
nnstudio/
  core/                        no Qt in here - pure domain logic
    dataset.py                 loading, task inference, encoding, splitting
    model_builder.py           NetworkConfig -> compiled Keras model
    trainer.py                 the training run, driven by plain callbacks
    crossval.py                k-fold estimate: k models trained, k discarded
    vision.py                  image datasets: synthetic shapes, CIFAR-10, folders
    resnet.py                  hand-built residual blocks + transfer learning
    vision_trainer.py          convolutional runs and the skip-connection ablation
  ui/
    main_window.py             thin shell: the two workspace tabs
    dense_workspace.py         the NN path: its 4 stages, visuals and state
    vision_workspace.py        the CNN path: its 4 stages, visuals and state
    data_panel.py              NN stage 1 - source, preview, adaptation
    architecture_panel.py      NN stage 2 - hidden stack, output head, optimiser
    training_panel.py          NN stage 3 - hyperparameters, k-fold, log
    predict_panel.py           NN stage 4 - manual inference
    vision/
      images_panel.py          CNN stage 1 - source and dataset build
      architecture_panel.py    CNN stage 2 - scratch stack or pretrained backbone
      training_panel.py        CNN stage 3 - run, compare arms, log
      predict_panel.py         CNN stage 4 - sample and read predictions
    network_canvas.py          the live diagram (QPainter)
    plots.py                   learning curves, fold bars, arm overlays
    resnet_canvas.py           stage chain plus one block opened up
    image_grid.py              image previews and prediction grids
    workers.py                 QThread adapters for training and cross-validation
    theme.py                   palette and stylesheet
    widgets.py                 shared building blocks
```

### How it is put together

One rule shapes the whole tree: **`core/` contains no Qt.** Everything in there
is plain Python that would run in a notebook or a script with no display — data
adaptation, model building, training runs. Everything in `ui/` is an adapter
over it. The two halves meet in exactly one file, `ui/workers.py`, which wraps a
plain run object in a `QThread` and turns its callbacks into signals.

That is why training never freezes the window, and why the interesting logic can
be checked without opening one.

The two workspaces are independent on purpose. They share the theme and the
plotting widgets and nothing else: no shared dataset, no shared model, no shared
worker. Training in one leaves the other untouched.

TensorFlow is imported lazily inside the training thread, so the window opens
immediately instead of waiting several seconds on startup.

## Contributing

Read [`AGENTS.md`](AGENTS.md) first. It lists the invariants that must not
break — most of them exist because a convenient shortcut would have made the app
report a flattering number or teach a wrong lesson.

There is no committed test suite yet; `AGENTS.md` documents how to verify a
change headlessly in the meantime, and adding a `tests/` folder is the single
most valuable contribution this repo could take.

Every number quoted in this README and in the manual was measured in this app,
not copied from a paper. If you change a default or touch the model code,
re-measure before editing the claim.
