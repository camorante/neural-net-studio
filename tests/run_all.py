"""Run every suite, cheapest first, and report what passed.

    python tests/run_all.py                 # everything
    python tests/run_all.py --fast          # skip the Qt suites
    python tests/run_all.py judge_gui core  # whatever matches those names

Each suite runs in its own process. They train real networks and build real Qt
applications, so sharing an interpreter between them would let one suite's
global state decide another suite's result.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Ordered by cost: a broken import should not cost four minutes to discover.
SUITES = (
    ("guards", "test_autoencoder_guards.py", False, "the readout refuses unearned lessons"),
    ("core", "test_autoencoder_core.py", False, "config, geometry, training, anomalies, sweep"),
    ("judge_core", "test_judge_core.py", False, "load an image from disk and score it"),
    ("seq_core", "test_sequence_core.py", False, "sequence tasks, floors and the order probe"),
    ("regression", "test_workspaces_regression.py", True, "dense and convolutional still train"),
    ("judge_gui", "test_judge_gui.py", True, "the judging buttons, end to end"),
    ("gui", "test_autoencoder_gui.py", True, "the whole autoencoder workspace"),
    ("seq_gui", "test_sequence_gui.py", True, "the whole sequence workspace"),
)


def chosen(argv: list[str]) -> list[tuple]:
    fast = "--fast" in argv
    names = [a for a in argv if not a.startswith("-")]
    picked = [s for s in SUITES if not (fast and s[2])]
    if not names:
        return picked
    # An exact name wins, so "gui" does not also drag in "judge_gui".
    exact = [s for s in picked if s[0] in names]
    return exact or [s for s in picked if any(n in s[0] for n in names)]


def main(argv: list[str]) -> int:
    suites = chosen(argv)
    if not suites:
        print("nothing matched", argv)
        return 2

    results, whole = [], time.monotonic()
    for name, filename, _qt, what in suites:
        print(f"\n{'=' * 72}\n== {name}: {what}\n{'=' * 72}", flush=True)
        started = time.monotonic()
        code = subprocess.run([sys.executable, str(HERE / filename)]).returncode
        results.append((name, code, time.monotonic() - started))

    print(f"\n{'=' * 72}")
    for name, code, took in results:
        print(f"  {'PASS' if code == 0 else 'FAIL':<5} {name:<12} {took:6.1f}s")
    failed = [n for n, code, _ in results if code != 0]
    print(f"  {len(results) - len(failed)}/{len(results)} passed "
          f"in {time.monotonic() - whole:.0f}s")
    if failed:
        print("  failing:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
