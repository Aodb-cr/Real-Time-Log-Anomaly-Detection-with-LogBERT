from __future__ import annotations

from pathlib import Path
import sys


# Ensure src/ is importable when running tests from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ..pipelines.detector import detect_anomalies, should_alert  # noqa: E402


def test_detect_anomalies_threshold():
    keys = [f"K{i}" for i in range(6)]
    probs = [0.5, 0.05, 0.2, 0.09, 0.15, 0.01]
    threshold = 0.1

    anomalies = detect_anomalies(keys, probs, threshold)
    idxs = [i for (i, _k, _p) in anomalies]

    assert len(anomalies) == 3
    assert idxs == [1, 3, 5]


def test_should_alert():
    assert should_alert(2, 2) is True
    assert should_alert(1, 2) is False
