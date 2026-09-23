"""The transformer workspace driven through its real widgets, offscreen.

Sections 6, 8 and 9 are the ones that matter: the same Inspect tab has to report
a faithful map, a decorative one and a distributed one, each on a model that
genuinely earned it, and the student has to be able to switch between the two
probe arms to see the decorative map for themselves.
"""
import time

import _bootstrap
_bootstrap.offscreen()

from PyQt6.QtWidgets import (  # noqa: E402 - must follow offscreen()
    QApplication, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox,
)
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402

import nnstudio.ui.transformer_workspace as tw  # noqa: E402
seen_dialogs = []
tw.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
tw.QMessageBox.information = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append((title, text)))
tw.QMessageBox.critical = staticmethod(
    lambda parent, title, text, *a, **k: seen_dialogs.append(("CRITICAL " + title, text)))

from nnstudio.core import transformer as tr  # noqa: E402
from nnstudio.ui.main_window import MainWindow, TRANSFORMER  # noqa: E402
from nnstudio.ui.theme import apply_theme  # noqa: E402

HERE = _bootstrap.output_dir()

app = QApplication([]); apply_theme(app)
win = MainWindow(); win.resize(1600, 960); win.show()
win.workspaces.setCurrentIndex(TRANSFORMER)
ws = win.transformer
dp, ap, tp, ip = (ws.data_panel, ws.architecture_panel,
                  ws.training_panel, ws.inspect_panel)
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
    dp.task_combo.setCurrentIndex(list(tr.TASKS).index(task))
    app.processEvents()


def build():
    ws._data_worker = None
    dp.build_button.click()
    assert ws._data_worker is not None, "the dataset worker never started"
    pump(ws._data_worker, 180_000)


def train(button):
    ws._train_worker = None
    button.click()
    assert ws._train_worker is not None, seen_dialogs
    pump(ws._train_worker)


print("=== 1. the shell has five workspaces ===")
titles = [win.workspaces.tabText(i) for i in range(win.workspaces.count())]
print("  tabs:", titles)
assert len(titles) == 5 and "Transformer" in titles[TRANSFORMER]
stages = [ws.tabs.tabText(i) for i in range(ws.tabs.count())]
assert stages == ["1. Data", "2. Architecture", "3. Training", "4. Inspect"], stages

print()
print("=== 2. the positional switch changes the diagram ===")
assert ws.canvas._positional and not ws.canvas._note
ap.positional_check.setChecked(False)
app.processEvents()
print("  off ->", ws.canvas._note[:66], "...")
assert not ws.canvas._positional and "OFF" in ws.canvas._note
assert "NO positional" in ws.diagram_card._title.text()
ap.positional_check.setChecked(True)
app.processEvents()
assert ws.canvas._positional and not ws.canvas._note

print()
print("=== 3. control heights survive the layout ===")
squeezed, checked = [], 0
for index, panel in enumerate((dp, ap, tp, ip)):
    ws.tabs.setCurrentIndex(index)
    app.processEvents()
    for widget in panel.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
        if not widget.isVisible():
            continue
        checked += 1
        if not 26 <= widget.height() < 400:
            squeezed.append((type(widget).__name__, widget.height(), stages[index]))
print("  visible controls measured:", checked, "| out of range:", squeezed or "none")
assert checked >= 12, f"only {checked} controls were laid out - the test is blind"
assert not squeezed, squeezed

print()
print("=== 4. picking a task rewrites its note ===")
for task in tr.TASKS:
    pick_task(task)
    assert dp.task_note.text() == tr.TASK_NOTES[task]
print("  all three notes follow the combo")
assert "TWO blocks" in tr.TASK_NOTES[tr.MATCH]

print()
print("=== 5. generate `match` through the real button ===")
pick_task(tr.MATCH)
build()
assert ws._bundle is not None, seen_dialogs
print("  summary:", dp.summary_label.text()[:74])
print("  floor -> Training:", tp.floor_label.text(), "| Architecture:", ap.floor_label.text())
assert tp.floor_label.text() in ap.floor_label.text()
assert ws.tabs.currentIndex() == 1

print()
print("=== 6. train, and the map is tested as well as drawn ===")
started = time.monotonic()
train(tp.train_button)
print(f"  trained in {time.monotonic() - started:.0f}s | val accuracy {tp.score_label.text()}"
      f" | vs floor {tp.versus_label.text()}")
log = tp.log.toPlainText()
assert "Testing the attention map" in log
assert "THIS MAP EARNED ITS READING" in log, log[-600:]
print("  verdict:", ip.verdict_label.text())
assert ip.verdict_label.text().startswith("Faithful")
assert ws.tabs.currentIndex() == 2
print("  blocks offered:", ip.block_spin.maximum(), "| showing block", ip.block_spin.value())
assert ip.block_spin.maximum() == 2 and ip.block_spin.value() == 2, \
    "the map shown first must be the block the verdict was measured on"
heads = [ip.head_combo.itemText(i) for i in range(ip.head_combo.count())]
print("  head choices:", heads)
assert heads == ["Average of all heads", "Head 1", "Head 2"]
# isVisibleTo, not isVisible: the Inspect tab is not in front here, so
# isVisible() is False for everything on it and this check would pass blind.
assert not ip.arm_combo.isVisibleTo(ip), "one run, one model - no arm to choose"

# The green column is where the answer lives; on a faithful map the most-weighed
# column should land on it for most sequences. Walk a few with the real button.
ws.tabs.setCurrentIndex(3); app.processEvents()
hits = 0
for _ in range(8):
    ip.next_button.click(); app.processEvents()
    hits += ws.attention.focus == ws.attention._expected
print(f"  most-attended column == answer column on {hits} of 8 sequences")
assert hits >= 6, hits
print("  caption:", ws.attention.caption)
print("  answer :", ip.answer_label.text())
assert "right answer is token" in ip.answer_label.text()
shot("tf_faithful.png")

print()
print("=== 7. a single head can be drawn on its own ===")
ip.head_combo.setCurrentIndex(1)
ip.show_button.click(); app.processEvents()
assert "head 1" in ws.attention.caption, ws.attention.caption
ip.block_spin.setValue(1)
ip.show_button.click(); app.processEvents()
assert "block 1" in ws.attention.caption
print("  ", ws.attention.caption)

print()
print("=== 8. the position probe, and the decorative map it leaves behind ===")
pick_task(tr.FIRST)
build()
started = time.monotonic()
train(tp.probe_button)
log = tp.log.toPlainText()
print(f"  probe took {time.monotonic() - started:.0f}s")
assert "POSITION IS ESSENTIAL HERE" in log, log[-500:]
arms = [ip.arm_combo.itemText(i) for i in range(ip.arm_combo.count())]
print("  arms offered:", arms, "| combo shown:", ip.arm_combo.isVisibleTo(ip))
assert arms == ["with positions", "no positions"] and ip.arm_combo.isVisibleTo(ip)
print("  with positions ->", ip.verdict_label.text())
assert ip.verdict_label.text().startswith("Faithful")
ip.arm_combo.setCurrentIndex(1); app.processEvents()
print("  no positions   ->", ip.verdict_label.text())
assert ip.verdict_label.text().startswith("Decoration"), ip.verdict_label.text()
assert "no positions" in ws.attention.caption
shot("tf_decorative.png")

print()
print("=== 9. a task where no single token matters ===")
pick_task(tr.MAJORITY)
build()
train(tp.train_button)
print("  verdict:", ip.verdict_label.text())
assert ip.verdict_label.text().startswith("No single token matters")
assert ws.attention._expected == -1, "majority has no answer column to mark"

print()
print("=== 10. Stop actually stops ===")
tp.epochs_spin.setValue(400)
ws._train_worker = None
tp.train_button.click()
worker = ws._train_worker
started = time.monotonic()
QTimer.singleShot(3000, tp.stop_button.click)
pump(worker, 180_000)
took = time.monotonic() - started
print(f"  a 400-epoch run ended {took:.1f}s in")
assert took < 60 and tp.train_button.isEnabled() and not tp.stop_button.isEnabled()

print()
print("=== 11. a new dataset invalidates the old model ===")
build()
assert not ws._arms and not ip.show_button.isEnabled()
assert ws.attention._weights is None
print("  model and map cleared")

print()
print("=== 12. the other four workspaces still build ===")
for index, name in ((0, "dense"), (1, "vision"), (2, "autoencoder"), (3, "sequence")):
    win.workspaces.setCurrentIndex(index); app.processEvents()
    assert getattr(win, name).tabs.count() == 4
print("  all four still have their four stages")

for title, text in seen_dialogs:
    print(f"  [{title}] {str(text)[:88]}")
assert not [d for d in seen_dialogs if d[0].startswith("CRITICAL")], seen_dialogs
win.close()
print()
print("TRANSFORMER GUI OK")
