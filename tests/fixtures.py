"""The hand-made images a student would feed to "Judge my own image".

They are drawn here rather than committed as binaries, so a fresh clone can run
the judging suites with nothing but the repository.

They are drawn at 256x256 on purpose: the dataset is 32x32, so every one of them
also exercises the resize inside load_single_image. The backdrop matches the one
make_synthetic paints, so the only thing that differs from a training image is
the shape itself - which is the whole point of the measurement.
"""
from __future__ import annotations

from pathlib import Path

BACKDROP = (12, 16, 28)


def _circle(draw, s):
    draw.ellipse([s * .3, s * .3, s * .7, s * .7], fill=(90, 220, 160))


def _square(draw, s):
    draw.rectangle([s * .3, s * .3, s * .7, s * .7], fill=(220, 150, 240))


def _triangle(draw, s):
    draw.polygon([(s * .5, s * .25), (s * .78, s * .72), (s * .22, s * .72)],
                 fill=(240, 120, 110))


def _stripes(draw, s):
    for top in range(0, s, s // 8):
        draw.rectangle([0, top, s, top + s // 16], fill=(250, 250, 90))


def _white(draw, s):
    draw.rectangle([0, 0, s, s], fill=(255, 255, 255))


# Ordered from "the model has seen thousands of these" to "as far away as a
# picture can get", which is the order the percentiles should come out in.
CASES = (
    ("my_circle.png", _circle, "a circle: the model knows these"),
    ("my_square.png", _square, "a square: it knows these too"),
    ("my_triangle.png", _triangle, "a TRIANGLE: it has never seen one"),
    ("my_stripes.png", _stripes, "stripes: nothing to do with the dataset"),
    ("my_white.png", _white, "solid white: as far from the data as it gets"),
)


def draw_case(directory, name, painter, size=256) -> Path:
    from PIL import Image, ImageDraw

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (size, size), BACKDROP)
    painter(ImageDraw.Draw(canvas), size)
    path = directory / name
    canvas.save(path)
    return path


def make_all(directory) -> list[tuple[Path, str]]:
    """Draw every case. Returns [(path, what it is), ...] in the order above."""
    return [(draw_case(directory, name, painter), what)
            for name, painter, what in CASES]
