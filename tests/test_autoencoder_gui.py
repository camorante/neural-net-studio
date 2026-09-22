"""The autoencoder workspace driven through its real widgets, offscreen.

Nothing is called directly that a student could not click. The point is that the
wiring is what breaks - a signal not connected, a button not re-enabled, a
metric never filled in - and none of that shows up in a core test.

Offscreen Qt has no Segoe UI, so WIDTH measurements come out roughly twice the
real ones and are worthless here. HEIGHT is reliable, which is why section 3
only checks heights.
"""
import time

import _bootstrap
_bootstrap.offscreen()

from PyQt6.QtWidgets import (  # noqa: E402 - must follow offscreen()
    QApplication, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox,
)
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402

# The confirm dialogs answer Yes; the informational ones are collected so an
# unexpected complaint cannot pass unnoticed.
import nnstudio.ui.autoencoder_workspace as aw  # noqa: E402
seen_dialogs = []
aw.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
aw.QMessageBox.information = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append((title, text)))
aw.QMessageBox.critical = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append(("CRITICAL " + title, text)))

from nnstudio.ui.main_window import MainWindow, AUTOENCODER  # noqa: E402
from nnstudio.ui.theme import apply_theme  # noqa: E402

HERE = _bootstrap.output_dir()

app = QApplication([]); apply_theme(app)
win = MainWindow(); win.resize(1600, 960); win.show()
win.workspaces.setCurrentIndex(AUTOENCODER)
ws = win.autoencoder
app.processEvents()


def shot(name):
    pix = QPixmap(win.size())
    win.render(pix)
    pix.save(str(HERE / name))


def pump(worker, timeout_ms=600_000):
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    app.processEvents()


print("=== 1. the autoencoder sits in its own workspace ===")
titles = [win.workspaces.tabText(i) for i in range(win.workspaces.count())]
print("  tabs:", titles)
assert "Autoencoder" in titles[AUTOENCODER]
stages = [ws.tabs.tabText(i) for i in range(ws.tabs.count())]
print("  stages:", stages)
assert stages == ["1. Images", "2. Architecture", "3. Training", "4. Reconstruct"]

print()
print("=== 2. the diagram exists before any data ===")
assert ws.canvas._columns, "canvas was never given an architecture"
print("  columns:", " -> ".join(c["name"] for c in ws.canvas._columns))
print("  caption:", ws.canvas._caption)
print("  note   :", (ws.canvas._note or "(none)")[:95])
assert "squeeze" in ws.canvas._caption
assert "MORE than" in ws.canvas._note, "the honest conv note is missing"

print()
print("=== 3. control heights survive the layout (the known trap) ===")
ap = ws.architecture_panel
squeezed, checked, unlaid = [], 0, []
for index, panel in enumerate(
    (ws.images_panel, ap, ws.training_panel, ws.reconstruct_panel)
):
    # A widget on a tab that was never shown reports its unlaid-out size, which
    # made an earlier version of this check blind. Switch to each tab first.
    ws.tabs.setCurrentIndex(index)
    app.processEvents()
    for widget in panel.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
        if not widget.isVisible():
            continue          # hidden page of a QStackedWidget: never laid out
        checked += 1
        height = widget.height()
        if height >= 400:
            unlaid.append((type(widget).__name__, height, stages[index]))
        elif height < 26:
            squeezed.append((type(widget).__name__, height, stages[index]))
print("  visible controls measured:", checked)
print("  never laid out (>=400px) :", unlaid or "none")
print("  squeezed below 26px      :", squeezed or "none")
assert checked >= 12, f"only {checked} controls were laid out - the test is blind"
assert not unlaid, unlaid
assert not squeezed, squeezed
ws.tabs.setCurrentIndex(1)
app.processEvents()
print("  latent slider height:", ap.latent_slider.height(), "(horizontal, expect < 40)")
assert 10 < ap.latent_slider.height() < 40, ap.latent_slider.height()

print()
print("=== 4. the latent slider drives the compression readout ===")
ap.latent_slider.setValue(2)
app.processEvents()
print("  latent 2  ->", ap.latent_label.text())
assert "1536x" in ap.latent_label.text()
ap.latent_spin.setValue(64)
app.processEvents()
print("  spin 64   ->", ap.latent_label.text())
assert ap.latent_slider.value() == 64, "the spin box did not drive the slider"
assert "48x" in ap.latent_label.text()
print("  problems  :", ap.problems() or "none")

print()
print("=== 5. a latent as wide as the image is refused ===")
ap.set_data_shape((16, 16, 1), locked=False)
ap.latent_spin.setValue(256)
app.processEvents()
print("  latent 256 of 256:", ap.problem_label.text()[:85])
assert "no bottleneck" in ap.problem_label.text()
ap.set_data_shape((32, 32, 3), locked=False)
ap.stages_spin.setValue(2)
ap.latent_spin.setValue(16)
app.processEvents()
assert not ap.problems()

print()
print("=== 6. build a dataset through the real button ===")
ip = ws.images_panel
ip.n_images_spin.setValue(300)
ip.n_shapes_spin.setValue(3)
ip.image_size_combo.setCurrentText("32")
ip.noise_spin.setValue(0.05)
ws._data_worker = None
ip.build_button.click()
assert ws._data_worker is not None, "the dataset worker never started"
pump(ws._data_worker, 180_000)
print("  summary:", ip.summary_label.text()[:82])
assert ws._bundle is not None, seen_dialogs
print("  moved to tab:", ws.tabs.currentIndex(), "(expect 1)")
assert ws.tabs.currentIndex() == 1
print("  hold-out choices:",
      [ap.anomaly_combo.itemText(i) for i in range(ap.anomaly_combo.count())])
assert ap.anomaly_combo.count() == 4
print("  give-up floor:", ws.training_panel.baseline_label.text()[:72])
assert "average" in ws.training_panel.baseline_label.text()
# The floor must read as a NUMBER in a tab, not only inside a sentence and on
# the chart - before the first epoch every other metric says "-".
print("  floor as a number -> Training:", tp0 := ws.training_panel.floor_label.text(),
      "| Architecture:", ap.floor_label.text())
assert tp0.startswith("0.0"), tp0
assert tp0 in ap.floor_label.text(), "both tabs must show the same value"
print("  original row drawn:", len(ws.grid._rows), "row(s)")
assert len(ws.grid._rows) == 1

print()
print("=== 7. train for real, unsupervised ===")
tp = ws.training_panel
tp.epochs_spin.setValue(4)
tp.batch_spin.setValue(32)
ap.latent_spin.setValue(16)
app.processEvents()
ws._train_worker = None
tp.train_button.click()
assert ws._train_worker is not None, seen_dialogs
print("  while training -> train:", tp.train_button.isEnabled(),
      "| stop:", tp.stop_button.isEnabled())
assert not tp.train_button.isEnabled() and tp.stop_button.isEnabled()
pump(ws._train_worker, 600_000)
assert ws._model is not None, seen_dialogs
print("  epoch:", tp.epoch_label.text(), "| val MSE:", tp.loss_label.text(),
      "| giving up:", tp.floor_label.text(),
      "| vs giving up:", tp.versus_label.text())
assert tp.versus_label.text() != "-", "the comparison metric never filled in"
assert tp.floor_label.text().startswith("0.0"), "reset_metrics wiped the floor"
assert ws.reconstruct_panel.reconstruct_button.isEnabled()
assert not ws.reconstruct_panel.anomaly_button.isEnabled(), "nothing was held out"
print("  grid rows:", [r["label"] for r in ws.grid._rows])
assert [r["label"] for r in ws.grid._rows] == ["original", "latent 16"]
assert ws.grid._rows[1]["errors"] is not None
print("  card:", ws.grid_card._title.text())
log = tp.log.toPlainText()
assert "No labels are used anywhere in this run" in log
assert "average image" in log or "give-up floor" in log, log[-300:]
print("  log tail:", log.strip().splitlines()[-1][:95])
shot("ae_trained.png")

print()
print("=== 8. hardest-to-rebuild ranking ===")
ws.reconstruct_panel.count_spin.setValue(6)
ws.reconstruct_panel.hardest_button.click()
app.processEvents()
print("  rows  :", [r["label"] for r in ws.grid._rows])
print("  head  :", ws.reconstruct_panel.result_label.text())
print("  detail:", ws.reconstruct_panel.detail_label.text()[:112])
assert len(ws.grid._rows) == 2
errs = list(ws.grid._rows[1]["errors"])
assert errs == sorted(errs, reverse=True), "not sorted worst first"

print()
print("=== 9. denoising shows the corrupted row it was actually fed ===")
ap.denoise_check.setChecked(True)
ap.noise_spin.setValue(0.30)
app.processEvents()
assert ap.noise_spin.isEnabled(), "the noise spin stayed disabled"
assert "denoising" in ws.canvas._caption, ws.canvas._caption
tp.epochs_spin.setValue(3)
ws._train_worker = None
tp.train_button.click()
pump(ws._train_worker, 600_000)
ws.reconstruct_panel.reconstruct_button.click()
app.processEvents()
labels = [r["label"] for r in ws.grid._rows]
print("  rows:", labels)
assert len(labels) == 3 and labels[1].startswith("corrupted"), labels
print("  caption:", ws.grid._caption[-70:])
assert "scored against the clean one" in ws.grid._caption
assert "Denoising run" in tp.log.toPlainText()
ap.denoise_check.setChecked(False)
app.processEvents()

print()
print("=== 10. holding a class out enables the anomaly score ===")
ap.anomaly_combo.setCurrentIndex(3)
held = ap.anomaly_combo.currentText()
print("  holding out:", held, "| index:", ap.anomaly_class())
assert ap.anomaly_class() == 2
tp.epochs_spin.setValue(4)
ws._train_worker = None
tp.train_button.click()
pump(ws._train_worker, 600_000)
assert ws._split is not None, "the split was never kept"
print("  split:", ws._split.describe())
assert ws._split.anomaly_name == held
assert ws.reconstruct_panel.anomaly_button.isEnabled()
ws.reconstruct_panel.anomaly_button.click()
app.processEvents()
print("  rows:", [r["label"] for r in ws.grid._rows])
assert len(ws.grid._rows) == 4
print("  head:", ws.reconstruct_panel.result_label.text())
assert "AUC" in ws.reconstruct_panel.result_label.text()
assert "ROC AUC" in tp.log.toPlainText()
shot("ae_anomaly.png")

print()
print("=== 11. the sweep preset and the sweep itself ===")
ap.anomaly_combo.setCurrentIndex(0)
app.processEvents()
ap.sweep_button.click()
app.processEvents()
print("  preset epochs/batch:", tp.epochs_spin.value(), "/", tp.batch_spin.value())
assert tp.epochs_spin.value() == 25
assert ws.tabs.currentIndex() == 2

tp.epochs_spin.setValue(3)   # deliberately too short: the readout must refuse
ws._train_worker = None
tp.sweep_button.click()
assert ws._train_worker is not None, seen_dialogs
pump(ws._train_worker, 900_000)
print("  arms:", [a["label"] for a in ws._sweep_arms])
assert len(ws._sweep_arms) == 3, seen_dialogs
labels = [r["label"] for r in ws.grid._rows]
print("  grid rows:", labels)
assert labels == ["original", "latent 2", "latent 8", "latent 64"], labels
assert len(ws._histories) == 3
assert "NO CONCLUSION IS AVAILABLE" in tp.log.toPlainText(), \
    "a 3-epoch sweep must not claim a lesson"
print("  3-epoch sweep refused to conclude: yes")
print("  card:", ws.grid_card._title.text())
shot("ae_sweep.png")

print()
print("=== 12. Stop actually stops ===")
tp.epochs_spin.setValue(200)
ws._train_worker = None
tp.train_button.click()
worker = ws._train_worker
started = time.monotonic()
QTimer.singleShot(4000, tp.stop_button.click)
pump(worker, 180_000)
took = time.monotonic() - started
print(f"  a 200-epoch run ended {took:.1f}s after it started")
assert tp.train_button.isEnabled() and not tp.stop_button.isEnabled()
ran = len(ws._histories.get("run", {}).get("loss", []))
print("  epochs completed:", ran, "of 200")
# The stop flag is checked on every BATCH, so it can land mid-epoch and no
# epoch_end fires at all. Either way the run must be nowhere near 200.
assert ran < 30, ran
assert took < 60, f"stop took {took:.1f}s - it is not being checked per batch"
assert ws._model is not None, "a stopped run should still hand back its model"

print()
print("=== 13. a new dataset invalidates the old model ===")
ip.n_images_spin.setValue(200)
ws._data_worker = None
ip.build_button.click()
pump(ws._data_worker, 180_000)
print("  model cleared  :", ws._model is None)
print("  reconstruct off:", not ws.reconstruct_panel.reconstruct_button.isEnabled())
print("  anomaly off    :", not ws.reconstruct_panel.anomaly_button.isEnabled())
assert ws._model is None and ws._split is None and not ws._sweep_arms
assert not ws.reconstruct_panel.reconstruct_button.isEnabled()
assert not ws.reconstruct_panel.anomaly_button.isEnabled()

print()
print("=== 14. the other workspaces still work ===")
win.workspaces.setCurrentIndex(0); app.processEvents()
print("  dense stages :", [win.dense.tabs.tabText(i)
                           for i in range(win.dense.tabs.count())])
win.workspaces.setCurrentIndex(1); app.processEvents()
print("  vision stages:", [win.vision.tabs.tabText(i)
                           for i in range(win.vision.tabs.count())])
assert win.vision.images_panel.build_button.isEnabled()
print("  shared source panel:", type(win.vision.images_panel).__name__,
      "/", type(ws.images_panel).__name__)
assert type(win.vision.images_panel).__name__ == "ImageSourcePanel"
assert type(ws.images_panel).__name__ == "ImageSourcePanel"
assert ws.images_panel is not win.vision.images_panel, "must be separate instances"

print()
print("=== dialogs raised during the run ===")
for title, text in seen_dialogs:
    print(f"  [{title}] {str(text)[:88]}")
assert not [d for d in seen_dialogs if d[0].startswith("CRITICAL")], seen_dialogs

win.close()
print()
print("AUTOENCODER GUI OK")
