"""The two older workspaces must still train after changes to the shared code.

Dense and convolutional share theme, widgets and the image source panel with the
autoencoder. This suite is the cheapest way to find out that a move or a rename
broke one of them.
"""
import _bootstrap
_bootstrap.offscreen()

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402

import nnstudio.ui.vision_workspace as vw  # noqa: E402
vw.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

from nnstudio.ui.main_window import MainWindow, DENSE, VISION  # noqa: E402
from nnstudio.ui.theme import apply_theme  # noqa: E402

app = QApplication([]); apply_theme(app)
win = MainWindow(); win.resize(1600, 960); win.show()


def pump(worker, ms=600_000):
    loop = QEventLoop(); worker.finished.connect(loop.quit)
    QTimer.singleShot(ms, loop.quit); loop.exec(); app.processEvents()


print("=== CNN workspace: build + train through the real buttons ===")
win.workspaces.setCurrentIndex(VISION); app.processEvents()
v = win.vision
v.images_panel.n_images_spin.setValue(200)
v.images_panel.n_shapes_spin.setValue(3)
v._data_worker = None
v.images_panel.build_button.click()
pump(v._data_worker, 180_000)
print("  dataset:", v.images_panel.summary_label.text()[:72])
assert v._bundle is not None

v.architecture_panel.stages_spin.setValue(2)
v.architecture_panel.filters_spin.setValue(8)
v.training_panel.epochs_spin.setValue(2)
app.processEvents()
v._train_worker = None
v.training_panel.train_button.click()
pump(v._train_worker, 600_000)
print("  trained :", v._model is not None,
      "| val acc:", v.training_panel.acc_label.text())
assert v._model is not None
v.predict_panel.predict_button.click(); app.processEvents()
print("  predict :", v.predict_panel.result_label.text()[:60])
assert "correct" in v.predict_panel.result_label.text()

print()
print("=== Dense workspace: prepare + train a built-in dataset ===")
win.workspaces.setCurrentIndex(DENSE); app.processEvents()
d = win.dense
names = [d.data_panel.builtin_combo.itemText(i)
         for i in range(d.data_panel.builtin_combo.count())]
print("  built-ins:", names[:4], "...")
d.data_panel.prepare_button.click(); app.processEvents()
print("  prepared :", d._bundle is not None)
assert d._bundle is not None
d.training_panel.epochs_spin.setValue(3)
d._worker = None
d.training_panel.train_button.click()
assert d._worker is not None, "dense train worker never started"
pump(d._worker, 300_000)
print("  trained  :", d._model is not None)
assert d._model is not None

win.close()
print()
print("REGRESSION OK - dense and convolutional both train")
