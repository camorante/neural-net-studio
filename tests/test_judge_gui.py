"""Load and judge your own image, through the real widgets.

The file chooser itself cannot be driven offscreen, so the path is assigned the
way QFileDialog would assign it and _refresh_judge is called - everything after
that point is the real button doing the real work.
"""
import _bootstrap
_bootstrap.offscreen()

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402

import nnstudio.ui.autoencoder_workspace as aw  # noqa: E402
seen = []
aw.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
aw.QMessageBox.information = staticmethod(lambda p, t, x, *a, **k: seen.append((t, x)))
aw.QMessageBox.critical = staticmethod(lambda p, t, x, *a, **k: seen.append(("CRIT " + t, x)))

import fixtures  # noqa: E402
from nnstudio.ui.main_window import MainWindow, AUTOENCODER  # noqa: E402
from nnstudio.ui.theme import apply_theme  # noqa: E402

HERE = _bootstrap.output_dir()
cases = fixtures.make_all(HERE)

app = QApplication([]); apply_theme(app)
win = MainWindow(); win.resize(1600, 960); win.show()
win.workspaces.setCurrentIndex(AUTOENCODER); app.processEvents()
ws = win.autoencoder
ip, ap, tp, rp = (ws.images_panel, ws.architecture_panel,
                  ws.training_panel, ws.reconstruct_panel)


def pump(worker, timeout_ms):
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    app.processEvents()


def choose(path):
    """What picking a file in the dialog leaves behind."""
    rp._path = str(path)
    rp.file_label.setText(str(path))
    rp._refresh_judge()
    app.processEvents()


print("=== 1. the button starts off and needs BOTH things ===")
print("   no model and no file:", rp.judge_button.isEnabled())
assert not rp.judge_button.isEnabled()
choose(HERE / "my_triangle.png")      # a file, but still no model
print("   file but no model   :", rp.judge_button.isEnabled(), "(must stay off)")
assert not rp.judge_button.isEnabled()

print()
print("=== 2. train with 'triangle' held out ===")
ip.n_images_spin.setValue(900); ip.noise_spin.setValue(0.05)
ws._data_worker = None; ip.build_button.click()
pump(ws._data_worker, 180_000)
ap.anomaly_combo.setCurrentIndex(3); ap.latent_spin.setValue(8)
tp.epochs_spin.setValue(25); app.processEvents()
ws._train_worker = None; tp.train_button.click()
pump(ws._train_worker, 600_000)
assert ws._model is not None, seen
print("   trained. val MSE:", tp.loss_label.text(), "| floor:", tp.floor_label.text())
choose(HERE / "my_triangle.png")
print("   button now:", rp.judge_button.isEnabled())
assert rp.judge_button.isEnabled(), "model + file should switch it on"

print()
print("=== 3. judge each file through the real button ===")
for path, _what in cases:
    choose(path)
    rp.judge_button.click(); app.processEvents()
    print(f"   {path.name:<18} -> {rp.result_label.text()}")
    assert len(ws.grid._rows) == 2, "should show your image and its rebuild"
    assert ws.grid._rows[0]["label"] == path.name[:14]
    assert ws.grid._rows[1]["label"] == "rebuilt"

print()
print("=== 4. the log carries the full readout and the latent code ===")
log = tp.log.toPlainText()
assert "Reconstruction error:" in log
assert "latent code" in log, "the latent code was never printed"
assert ("TRUST THIS ONE LOOSELY" in log or "emphatic end" in log), \
    "the seed-stability caveat is missing"
for line in [l for l in log.splitlines() if l.strip()][-6:]:
    print("   ", line[:96])

print()
print("=== 5. invalid files do not take the app down ===")
for path, why in ((HERE / "does_not_exist.png", "missing"),
                  (__file__, "not an image")):
    choose(path)
    rp.judge_button.click(); app.processEvents()
    print(f"   {why:<12} -> {rp.result_label.text()} | {rp.detail_label.text()[:54]}")
    assert "Could not judge" in rp.result_label.text()

print()
print("=== 6. a new dataset switches the button off again ===")
ws._data_worker = None; ip.build_button.click()
pump(ws._data_worker, 180_000)
print("   model:", ws._model, "| button:", rp.judge_button.isEnabled())
assert ws._model is None and not rp.judge_button.isEnabled()

print()
print("=== 7. heights of the new controls ===")
ws.tabs.setCurrentIndex(3); app.processEvents()
for name, widget in (("Choose...", rp.choose_button), ("Judge", rp.judge_button),
                     ("count spin", rp.count_spin)):
    print(f"   {name:<11} height {widget.height()}px")
    assert widget.height() >= 26, f"{name} is squeezed"

pix = QPixmap(win.size()); win.render(pix); pix.save(str(HERE / "tab_judge.png"))
print()
print("unexpected dialogs:", [d[0] for d in seen] or "none")
assert not [d for d in seen if d[0].startswith("CRIT")], seen
win.close()
print()
print("JUDGE GUI OK")
