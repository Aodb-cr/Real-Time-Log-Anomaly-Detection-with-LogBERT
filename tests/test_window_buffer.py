from __future__ import annotations

from pathlib import Path
import sys


# Ensure src/ is importable when running tests from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ..pipelines.window_buffer import SlidingWindowBuffer  # noqa: E402


def test_sliding_window_keeps_last_n():
    win = SlidingWindowBuffer(window_size=5)
    lines = [f"line-{i}" for i in range(8)]  # 0..7
    last_window = []
    for ln in lines:
        last_window = win.add(ln)

    assert win.size() == 5
    assert last_window == [f"line-{i}" for i in range(3, 8)]
    # snapshot returns the same content
    assert win.snapshot() == last_window
