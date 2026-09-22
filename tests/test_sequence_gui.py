"""The sequence workspace driven through its real widgets, offscreen.

Nothing here is called that a student could not click. As with the autoencoder
suite, only HEIGHT is measured on the controls - offscreen Qt has no Segoe UI,
so widths come out roughly twice the real ones.
"""
import time

import _bootstrap
_bootstrap.offscreen()

from PyQt6.QtWidgets import (  # noqa: E402 - must follow offscreen()
    QApplication, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox,
)
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402

import nnstudio.ui.sequence_workspace as sw  # noqa: E402
seen_dialogs = []
sw.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
sw.QMessageBox.information = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append((title, text)))
sw.QMessageBox.critical = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append(("CRITICAL " + title, text)))

from nnstudio.core import sequences as sq  # noqa: E402
from nnstudio.core.sequence_trainer import SWEEP_KINDS  # noqa: E402
from nnstudio.ui.main_window import MainWindow, SEQUENCE  # noqa: E402
from nnstudio.ui.theme import apply_theme  # noqa: E402

HERE = _bootstrap.output_dir()

app = QApplication([]); apply_theme(app)
win = MainWindow(); win.resize(1600, 960); win.show()
win.workspaces.setCurrentIndex(SEQUENCE)
ws = win.sequence
dp, ap, tp, pp = (ws.data_panel, ws.architecture_panel,
                  ws.training_panel, ws.predict_panel)
app.processEvents()


def shot(name):
    pix = QPixmap(win.size())
    win.render(pix)
    pix.save(str(HERE / name))


def pump(worker, timeout_ms=900_000):
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    app.processEvents()


def pick_task(task):
    dp.task_combo.setCurrentIndex(list(sq.TASKS).index(task))
    app.processEvents()


def build():
    ws._data_worker = None
    dp.build_button.click()
    assert ws._data_worker is not None, "the dataset worker never started"
    pump(ws._data_worker, 180_000)


print("=== 1. the shell has four workspaces ===")
titles = [win.workspaces.tabText(i) for i in range(win.workspaces.count())]
print("  tabs:", titles)
assert len(titles) == 4 and "Sequences" in titles[3]
stages = [ws.tabs.tabText(i) for i in range(ws.tabs.count())]
print("  stages:", stages)
assert stages == ["1. Data", "2. Architecture", "3. Training", "4. Inspect"]

print()
print("=== 2. the diagram exists before any data, with the state loop ===")
assert ws.canvas._columns, "canvas was never given an architecture"
print("  columns:", " -> ".join(c["name"] for c in ws.canvas._columns))
print("  recurrent:", ws.canvas._recurrent, "| length:", ws.canvas._length)
assert ws.canvas._recurrent, "LSTM is the default and must draw its state loop"
assert ws.canvas._length == dp.planned_length()

print()
print("=== 3. control heights survive the layout ===")
squeezed, checked, unlaid = [], 0, []
for index, panel in enumerate((dp, ap, tp, pp)):
    ws.tabs.setCurrentIndex(index)
    app.processEvents()
    for widget in panel.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
        if not widget.isVisible():
            continue
        checked += 1
        height = widget.height()
        if height >= 400:
            unlaid.append((type(widget).__name__, height, stages[index]))
        elif height < 26:
            squeezed.append((type(widget).__name__, height, stages[index]))
print("  visible controls measured:", checked)
print("  never laid out (>=400px) :", unlaid or "none")
print("  squeezed below 26px      :", squeezed or "none")
assert checked >= 10, f"only {checked} controls were laid out - the test is blind"
assert not unlaid, unlaid
assert not squeezed, squeezed

print()
print("=== 4. picking a task rewrites its note and the head ===")
for task in sq.TASKS:
    pick_task(task)
    note = dp.task_note.text()
    print(f"  {task:<6} outputs {ap._n_outputs} | note: {note[:56]}...")
    assert note == sq.TASK_NOTES[task]
    assert ap._task_kind == sq.TASK_KINDS[task]
    assert ap._n_outputs == (2 if sq.TASK_KINDS[task] == sq.CLASSIFY else 1)

print()
print("=== 5. bidirectional only offered where it means something ===")
for kind, expect in ((sq.LSTM, True), (sq.CONV1D, False), (sq.DENSE, False)):
    ap.kind_combo.setCurrentIndex(list(sq.KINDS).index(kind))
    app.processEvents()
    print(f"  {kind:<7} enabled {ap.bidirectional_check.isEnabled()} | "
          f"loop drawn {ws.canvas._recurrent}")
    assert ap.bidirectional_check.isEnabled() is expect
    assert ws.canvas._recurrent is (kind in (sq.LSTM, sq.GRU, sq.RNN))
    if not expect:
        assert not ap.bidirectional_check.isChecked()
        assert "no state carried" in (ws.canvas._note or "")
assert not ap.problems(), ap.problems()

print()
print("=== 6. generate a dataset through the real button ===")
pick_task(sq.ORDER)
ap.kind_combo.setCurrentIndex(list(sq.KINDS).index(sq.LSTM))
dp.count_spin.setValue(800)
dp.length_spin.setValue(16)
app.processEvents()
build()
print("  summary:", dp.summary_label.text()[:76])
assert ws._bundle is not None, seen_dialogs
print("  moved to tab:", ws.tabs.currentIndex(), "(expect 1)")
assert ws.tabs.currentIndex() == 1
print("  floor -> Training:", floor := tp.floor_label.text(),
      "| Architecture:", ap.floor_label.text())
assert floor.startswith("0."), floor
assert floor in ap.floor_label.text(), "both tabs must show the same value"
assert "commonest class" in tp.baseline_label.text(), tp.baseline_label.text()
print("  windows plotted:", ws.series.caption[:58])
assert "Five windows" in ws.series.caption

print()
print("=== 7. train for real ===")
tp.epochs_spin.setValue(8)
ws._train_worker = None
tp.train_button.click()
assert ws._train_worker is not None, seen_dialogs
assert not tp.train_button.isEnabled() and tp.stop_button.isEnabled()
pump(ws._train_worker)
assert ws._model is not None, seen_dialogs
print("  epoch:", tp.epoch_label.text(), "| validation:", tp.score_label.text(),
      "| floor:", tp.floor_label.text(), "| vs:", tp.versus_label.text())
assert tp.versus_label.text().endswith("pts"), tp.versus_label.text()
assert tp.floor_label.text().startswith("0."), "reset_metrics wiped the floor"
# A classification task offers the confusion view and hides the forecast one.
print("  confusion on:", pp.confusion_button.isEnabled(),
      "| forecast on:", pp.forecast_button.isEnabled())
assert pp.confusion_button.isEnabled() and not pp.forecast_button.isEnabled()
print("  auto-shown:", ws.series.caption)
assert "correct" in ws.series.caption
print("  result:", pp.result_label.text())
assert "majority class" in tp.log.toPlainText()
shot("seq_trained.png")

print()
print("=== 8. the order probe answers the question ===")
ws._train_worker = None
tp.probe_button.click()
assert ws._train_worker is not None, seen_dialogs
pump(ws._train_worker)
log = tp.log.toPlainText()
print("  arms:", [a["label"] for a in ws._arms])
assert [a["label"] for a in ws._arms] == ["real order", "time shuffled"]
assert "ORDER MATTERS HERE" in log, log[-400:]
print("  verdict: ORDER MATTERS HERE")
assert len(ws._histories) == 2
shot("seq_probe.png")

print()
print("=== 9. the same probe on a task where order is irrelevant ===")
pick_task(sq.COUNT)
build()
tp.epochs_spin.setValue(10)
ws._train_worker = None
tp.probe_button.click()
pump(ws._train_worker)
log = tp.log.toPlainText()
assert "ORDER CARRIED NOTHING" in log, log[-400:]
print("  verdict: ORDER CARRIED NOTHING - the same probe, the opposite answer")

print()
print("=== 10. the architecture sweep ===")
ap.sweep_button.click()
app.processEvents()
print("  preset epochs/batch:", tp.epochs_spin.value(), "/", tp.batch_spin.value())
assert tp.epochs_spin.value() == 20 and ws.tabs.currentIndex() == 2
tp.epochs_spin.setValue(5)
ws._train_worker = None
tp.sweep_button.click()
assert ws._train_worker is not None, seen_dialogs
pump(ws._train_worker)
print("  arms:", [a["label"] for a in ws._arms])
assert [a["label"] for a in ws._arms] == list(SWEEP_KINDS)
assert len(ws._histories) == 4
print("  params per arm:", {a["label"]: a["trainable_params"] for a in ws._arms})
shot("seq_sweep.png")

print()
print("=== 11. a forecasting task swaps the views over ===")
pick_task(sq.WALK)
build()
tp.epochs_spin.setValue(4)
ws._train_worker = None
tp.train_button.click()
pump(ws._train_worker)
print("  forecast on:", pp.forecast_button.isEnabled(),
      "| confusion on:", pp.confusion_button.isEnabled())
assert pp.forecast_button.isEnabled() and not pp.confusion_button.isEnabled()
print("  chart:", ws.series.caption)
assert "persistence" in ws.series.caption, "persistence must be on the chart"
print("  result:", pp.result_label.text())
assert "persistence" in pp.result_label.text()
# Losing must read as "3.9x WORSE", never as "0.25x worse".
assert "0.2" not in pp.result_label.text().split("(")[-1], pp.result_label.text()
assert "WORSE" in pp.result_label.text(), pp.result_label.text()
assert "DID NOT BEAT PERSISTENCE" in tp.log.toPlainText(), \
    "a random walk cannot be beaten and the log must say so"
shot("seq_forecast.png")

print()
print("=== 12. Stop actually stops ===")
tp.epochs_spin.setValue(300)
ws._train_worker = None
tp.train_button.click()
worker = ws._train_worker
started = time.monotonic()
QTimer.singleShot(3000, tp.stop_button.click)
pump(worker, 180_000)
took = time.monotonic() - started
ran = len(ws._histories.get("run", {}).get("loss", []))
print(f"  a 300-epoch run ended {took:.1f}s in, after {ran} epochs")
assert tp.train_button.isEnabled() and not tp.stop_button.isEnabled()
assert ran < 60, ran
assert took < 60, f"stop took {took:.1f}s - it is not checked per batch"

print()
print("=== 13. a new dataset invalidates the old model ===")
build()
print("  model cleared:", ws._model is None,
      "| views off:", not pp.forecast_button.isEnabled())
assert ws._model is None and not ws._arms
assert not pp.forecast_button.isEnabled()
assert not pp.confusion_button.isEnabled()

print()
print("=== 14. the other three workspaces still work ===")
for index, name in ((0, "dense"), (1, "vision"), (2, "autoencoder")):
    win.workspaces.setCurrentIndex(index)
    app.processEvents()
    panel = getattr(win, name)
    print(f"  {name:<12}", [panel.tabs.tabText(i) for i in range(panel.tabs.count())])
    assert panel.tabs.count() == 4

print()
print("=== dialogs raised during the run ===")
for title, text in seen_dialogs:
    print(f"  [{title}] {str(text)[:88]}")
assert not [d for d in seen_dialogs if d[0].startswith("CRITICAL")], seen_dialogs

win.close()
print()
print("SEQUENCE GUI OK")
