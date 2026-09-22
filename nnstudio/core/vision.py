"""Image datasets for the convolutional tab.

Images are kept as float32 in the 0..255 range. Normalisation lives inside the
model as a layer, so every path carries its own preprocessing and the preview
grid can render the raw pixels without undoing anything.
"""
from __future__ import annotations

import hashlib
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SHAPE_NAMES = ("circle", "square", "triangle", "cross")

SYNTHETIC = "synthetic"
CIFAR10 = "cifar10"
FOLDER = "folder"

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


class VisionError(RuntimeError):
    """Raised when an image dataset cannot be built or loaded."""


class BuildCancelled(VisionError):
    """Raised when the user stops a download or a long build."""


# The CIFAR-10 archive comes from a university server that serves it at roughly
# 0.1 MB/s, so a 170 MB transfer takes around half an hour. Keras downloads it
# with no cancel and no resume: interrupt it and the next attempt starts from
# zero. We fetch it ourselves - with progress, resume and cancel - and drop the
# verified archive exactly where Keras looks, so load_data() finds it cached.
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CIFAR10_ARCHIVE_NAME = "cifar-10-batches-py-target_archive"
CIFAR10_TOTAL_BYTES = 170_498_071

_CHUNK = 128 * 1024


@dataclass
class ImageBundle:
    """A ready-to-train image dataset, already split."""

    name: str
    source: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    class_names: list

    @property
    def image_size(self) -> tuple:
        return int(self.x_train.shape[1]), int(self.x_train.shape[2])

    @property
    def channels(self) -> int:
        return int(self.x_train.shape[3])

    @property
    def input_shape(self) -> tuple:
        return tuple(int(v) for v in self.x_train.shape[1:])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.x_val.shape[0])

    def describe(self) -> str:
        h, w = self.image_size
        return (
            f"{self.name}: {self.n_train} training / {self.n_val} validation images, "
            f"{h}x{w}x{self.channels}, {self.n_classes} classes "
            f"({', '.join(self.class_names)})"
        )

    def sample(self, count: int = 12, rng=None) -> tuple:
        """A small batch of validation images plus their true class indices."""
        rng = rng or np.random.default_rng()
        total = self.n_val if self.n_val else self.n_train
        source_x = self.x_val if self.n_val else self.x_train
        source_y = self.y_val if self.n_val else self.y_train
        picks = rng.choice(total, size=min(count, total), replace=False)
        return source_x[picks], np.argmax(source_y[picks], axis=1), picks


# ------------------------------------------------------------------- synthetic

@dataclass
class SyntheticSpec:
    """Knobs for the offline shape generator."""

    n_images: int = 1500
    image_size: int = 32
    n_classes: int = 3
    noise: float = 0.10
    rotate: bool = True
    vary_position: bool = True
    vary_size: bool = True
    val_split: float = 0.2
    seed: int = 7


def _polygon(kind: str, cx: float, cy: float, radius: float, angle: float) -> list:
    """Vertices for one shape, already rotated around its centre."""
    if kind == "square":
        offsets = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    elif kind == "triangle":
        offsets = [
            (math.cos(math.radians(a)), math.sin(math.radians(a)))
            for a in (-90, 30, 150)
        ]
    elif kind == "cross":
        t = 0.38
        offsets = [
            (-t, -1), (t, -1), (t, -t), (1, -t), (1, t), (t, t),
            (t, 1), (-t, 1), (-t, t), (-1, t), (-1, -t), (-t, -t),
        ]
    else:
        raise VisionError(f"Not a polygon shape: {kind}")

    points = []
    for ox, oy in offsets:
        rx = ox * math.cos(angle) - oy * math.sin(angle)
        ry = ox * math.sin(angle) + oy * math.cos(angle)
        points.append((cx + rx * radius, cy + ry * radius))
    return points


def make_synthetic(spec: SyntheticSpec, on_progress=None, should_stop=None) -> ImageBundle:
    """Draw coloured shapes at random positions, sizes and angles.

    Random placement is the point: it is what makes translation invariance
    worth having, and therefore what a convolution buys you over a dense layer.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415 - optional at import time

    if not 2 <= spec.n_classes <= len(SHAPE_NAMES):
        raise VisionError(f"Pick between 2 and {len(SHAPE_NAMES)} shape classes")
    if spec.n_images < 40:
        raise VisionError("Generate at least 40 images")

    rng = np.random.default_rng(spec.seed)
    classes = list(SHAPE_NAMES[: spec.n_classes])
    size = int(spec.image_size)
    scale = 4  # draw big, shrink down: cheap antialiasing

    images = np.zeros((spec.n_images, size, size, 3), dtype="float32")
    labels = np.zeros(spec.n_images, dtype="int64")

    every = max(1, spec.n_images // 100)
    for index in range(spec.n_images):
        if index % every == 0:
            _check(should_stop)
            _report(on_progress, index, spec.n_images, "Drawing shapes")
        kind_index = int(rng.integers(0, len(classes)))
        kind = classes[kind_index]
        labels[index] = kind_index

        canvas = Image.new("RGB", (size * scale, size * scale), (12, 16, 28))
        draw = ImageDraw.Draw(canvas)

        radius_frac = rng.uniform(0.22, 0.38) if spec.vary_size else 0.30
        radius = radius_frac * size * scale
        margin = radius * 1.15
        if spec.vary_position:
            cx = rng.uniform(margin, size * scale - margin)
            cy = rng.uniform(margin, size * scale - margin)
        else:
            cx = cy = size * scale / 2.0
        angle = rng.uniform(0, 2 * math.pi) if spec.rotate else 0.0

        colour = tuple(int(c) for c in rng.integers(120, 256, size=3))
        if kind == "circle":
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius], fill=colour
            )
        else:
            draw.polygon(_polygon(kind, cx, cy, radius, angle), fill=colour)

        frame = np.asarray(
            canvas.resize((size, size), Image.LANCZOS), dtype="float32"
        )
        if spec.noise > 0:
            frame = frame + rng.normal(0, spec.noise * 255.0, frame.shape)
        images[index] = np.clip(frame, 0.0, 255.0)

    _report(on_progress, spec.n_images, spec.n_images, "Splitting the dataset")
    return _split(
        images, labels, classes, spec.val_split, spec.seed,
        name=f"Shapes ({', '.join(classes)})", source=SYNTHETIC,
    )


# -------------------------------------------------------------------- download

def _report(on_progress, done: int, total: int, label: str, speed: float = 0.0) -> None:
    if on_progress:
        on_progress(int(done), int(total), label, float(speed))


def _check(should_stop) -> None:
    if should_stop and should_stop():
        raise BuildCancelled("Cancelled")


def cifar10_archive_path() -> Path:
    """Where Keras expects the downloaded CIFAR-10 archive to sit."""
    return Path.home() / ".keras" / "datasets" / CIFAR10_ARCHIVE_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_cifar10_archive(on_progress=None, should_stop=None) -> Path:
    """Put a verified CIFAR-10 archive in the Keras cache, resuming if possible.

    Downloads into a `.part` file and only renames it once the SHA-256 matches,
    so an interrupted transfer can never be mistaken for a complete one. The
    server supports range requests, so cancelling and coming back later picks up
    where it stopped instead of starting over.
    """
    target = cifar10_archive_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")

    if target.exists():
        _report(on_progress, 0, 1, "Checking the cached archive...")
        if _sha256(target) == CIFAR10_SHA256:
            _report(on_progress, 1, 1, "CIFAR-10 already downloaded")
            return target
        # A leftover from an interrupted Keras download. It is not resumable
        # because we cannot tell how much of it is valid, so it goes.
        target.unlink()

    resume_from = partial.stat().st_size if partial.exists() else 0
    if resume_from >= CIFAR10_TOTAL_BYTES:
        resume_from = 0
        partial.unlink()

    headers = {"User-Agent": "Mozilla/5.0 (Neural Net Studio)"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(CIFAR10_URL, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if resume_from and exc.code in (416, 501):  # range refused - start over
            partial.unlink(missing_ok=True)
            return ensure_cifar10_archive(on_progress, should_stop)
        raise VisionError(f"Could not reach the CIFAR-10 server: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise VisionError(f"Could not reach the CIFAR-10 server: {exc}") from exc

    with response:
        ranged = response.status == 206
        if resume_from and not ranged:
            resume_from = 0  # server ignored the range, restart cleanly
        declared = int(response.headers.get("Content-Length") or 0)
        total = (resume_from + declared) if declared else CIFAR10_TOTAL_BYTES

        done = resume_from
        started = time.monotonic()
        started_at = done
        mode = "ab" if resume_from else "wb"
        label = "Downloading CIFAR-10"

        _report(on_progress, done, total, label)
        try:
            with partial.open(mode) as handle:
                while True:
                    _check(should_stop)
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    elapsed = time.monotonic() - started
                    speed = (done - started_at) / elapsed if elapsed > 0.5 else 0.0
                    _report(on_progress, done, total, label, speed)
        except BuildCancelled:
            # The partial file stays put so the next attempt resumes from here.
            raise

    if done < total:
        raise VisionError(
            f"The download stopped early at {done / 1e6:.1f} MB of {total / 1e6:.1f} MB. "
            "Press Build dataset again to resume from where it left off."
        )

    _report(on_progress, total, total, "Verifying the archive...")
    if _sha256(partial) != CIFAR10_SHA256:
        partial.unlink(missing_ok=True)
        raise VisionError(
            "The downloaded archive is corrupt and was deleted. Try again."
        )
    partial.replace(target)
    return target


# ----------------------------------------------------------------------- cifar

def load_cifar10(
    class_names: list,
    per_class: int = 500,
    val_split: float = 0.2,
    seed: int = 7,
    image_size: int = 32,
    on_progress=None,
    should_stop=None,
) -> ImageBundle:
    """A CPU-sized slice of CIFAR-10. Downloads ~170 MB the first time."""
    ensure_cifar10_archive(on_progress, should_stop)
    _check(should_stop)
    _report(on_progress, 0, 0, "Unpacking and loading CIFAR-10...")

    from tensorflow import keras  # noqa: PLC0415 - lazy

    unknown = [c for c in class_names if c not in CIFAR10_CLASSES]
    if unknown:
        raise VisionError(f"Not CIFAR-10 classes: {', '.join(unknown)}")
    if len(class_names) < 2:
        raise VisionError("Pick at least 2 classes")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test]).astype("float32")
    y = np.concatenate([y_train, y_test]).reshape(-1)

    rng = np.random.default_rng(seed)
    kept_x, kept_y = [], []
    for new_index, name in enumerate(class_names):
        original = CIFAR10_CLASSES.index(name)
        rows = np.flatnonzero(y == original)
        picks = rng.choice(rows, size=min(per_class, rows.size), replace=False)
        kept_x.append(x[picks])
        kept_y.append(np.full(picks.size, new_index, dtype="int64"))

    images = np.concatenate(kept_x)
    labels = np.concatenate(kept_y)
    if image_size != 32:
        images = _resize(images, image_size)
    return _split(
        images, labels, list(class_names), val_split, seed,
        name=f"CIFAR-10 ({', '.join(class_names)})", source=CIFAR10,
    )


# ---------------------------------------------------------------------- folder

def load_folder(
    root: str | Path,
    image_size: int = 32,
    val_split: float = 0.2,
    seed: int = 7,
    max_per_class: int = 800,
    on_progress=None,
    should_stop=None,
) -> ImageBundle:
    """One subfolder per class, images inside. Nothing is downloaded."""
    from PIL import Image  # noqa: PLC0415

    base = Path(root)
    if not base.is_dir():
        raise VisionError(f"{base} is not a folder")

    class_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise VisionError(
            f"{base.name} needs at least 2 subfolders, one per class "
            "(for example cats/ and dogs/)"
        )

    rng = np.random.default_rng(seed)
    frames, labels, names = [], [], []
    expected = sum(
        len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        for d in class_dirs
    )
    for index, folder in enumerate(class_dirs):
        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            continue
        if len(files) > max_per_class:
            files = [files[i] for i in rng.choice(len(files), max_per_class, False)]
        loaded = 0
        for path in files:
            if len(frames) % 25 == 0:
                _check(should_stop)
                _report(on_progress, len(frames), expected,
                        f"Reading {folder.name}")
            try:
                with Image.open(path) as handle:
                    rgb = handle.convert("RGB").resize(
                        (image_size, image_size), Image.LANCZOS
                    )
                    frames.append(np.asarray(rgb, dtype="float32"))
            except Exception:  # noqa: BLE001 - a broken file is not fatal
                continue
            labels.append(index)
            loaded += 1
        if loaded:
            names.append(folder.name)

    if len(names) < 2:
        raise VisionError("Fewer than 2 classes had readable images")
    if len(frames) < 40:
        raise VisionError(f"Only {len(frames)} images loaded - at least 40 are needed")

    # Class indices must stay dense after skipping empty folders.
    remap = {old: new for new, old in enumerate(sorted(set(labels)))}
    labels = np.array([remap[v] for v in labels], dtype="int64")
    return _split(
        np.stack(frames), labels, names, val_split, seed,
        name=f"{base.name} ({len(names)} classes)", source=FOLDER,
    )


# ------------------------------------------------------------- single image

def load_single_image(path: str | Path, image_size: int,
                      channels: int = 3) -> np.ndarray:
    """One image from disk, reshaped to match a dataset. Returns (1, H, W, C).

    This is the inference counterpart to `load_folder`: no labels, no split,
    one picture. It lives here rather than in a workspace because asking a
    trained model about a file you just picked is not specific to any kind of
    network - the convolutional tab has the same gap.
    """
    from PIL import Image  # noqa: PLC0415

    target = Path(path)
    if not target.is_file():
        raise VisionError(f"{target} is not a file")
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        raise VisionError(
            f"{target.suffix or 'that file'} is not a readable image. "
            f"Use one of: {', '.join(IMAGE_SUFFIXES)}"
        )

    try:
        with Image.open(target) as handle:
            frame = handle.convert("RGB" if channels == 3 else "L").resize(
                (int(image_size), int(image_size)), Image.LANCZOS
            )
            array = np.asarray(frame, dtype="float32")
    except VisionError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise VisionError(f"Could not read {target.name}: {exc}") from exc

    if array.ndim == 2:
        array = array[..., None]
    return array[None, ...]


# --------------------------------------------------------------------- helpers

def _resize(images: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    out = np.zeros((len(images), size, size, images.shape[3]), dtype="float32")
    for i, frame in enumerate(images):
        picture = Image.fromarray(frame.astype("uint8"))
        out[i] = np.asarray(
            picture.resize((size, size), Image.LANCZOS), dtype="float32"
        )
    return out


def _split(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    val_split: float,
    seed: int,
    *,
    name: str,
    source: str,
) -> ImageBundle:
    """Stratified split plus one-hot encoding."""
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    val_split = float(min(max(val_split, 0.05), 0.5))
    counts = np.bincount(labels, minlength=len(class_names))
    stratify = labels if counts.min() >= 2 else None
    train_idx, val_idx = train_test_split(
        np.arange(len(images)),
        test_size=val_split,
        random_state=seed,
        stratify=stratify,
    )
    one_hot = np.eye(len(class_names), dtype="float32")[labels]
    return ImageBundle(
        name=name,
        source=source,
        x_train=images[train_idx],
        y_train=one_hot[train_idx],
        x_val=images[val_idx],
        y_val=one_hot[val_idx],
        class_names=list(class_names),
    )
