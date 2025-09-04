from __future__ import annotations

"""Simple anomaly detection helpers for LogBERT probabilities.

Functions
- detect_anomalies(sequence_keys, probs, threshold):
    Returns a list of (index, original_key, probability) where probability < threshold.

- should_alert(anomaly_count, alert_threshold):
    Returns True when the anomaly count meets or exceeds the alert threshold.

Tests (illustrative)
--------------------
>>> detect_anomalies(["A","B","C"], [0.20, 0.05, 0.90], 0.10)
[(1, 'B', 0.05)]
>>> should_alert(2, 2)
True
>>> should_alert(1, 2)
False
"""

from typing import List, Tuple


def detect_anomalies(
    sequence_keys: list[str],
    probs: list[float],
    threshold: float,
) -> list[tuple[int, str, float]]:
    """Identify anomalous events by probability threshold.

    Parameters
    - sequence_keys: normalized log keys (templates) in sequence order
    - probs: probability scores aligned with sequence_keys
    - threshold: probability cutoff; any p < threshold is flagged as anomaly

    Returns
    - List of (index, original_key, probability) for anomalies

    Raises
    - ValueError on length mismatch or invalid threshold
    """
    if len(sequence_keys) != len(probs):
        raise ValueError(
            f"Length mismatch: sequence_keys={len(sequence_keys)} probs={len(probs)}"
        )
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be within [0.0, 1.0]")

    anomalies: List[Tuple[int, str, float]] = []
    for idx, (key, p) in enumerate(zip(sequence_keys, probs)):
        try:
            pf = float(p)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Invalid probability at index {idx}: {p!r}") from e
        if pf < threshold:
            anomalies.append((idx, key, pf))
    return anomalies


def should_alert(anomaly_count: int, alert_threshold: int) -> bool:
    """Decide if an alert should be triggered.

    Returns True when anomaly_count >= alert_threshold.

    Examples
    >>> should_alert(3, 2)
    True
    >>> should_alert(1, 2)
    False
    """
    if alert_threshold <= 0:
        # Treat non-positive threshold as "always alert" guardrail
        return True
    return int(anomaly_count) >= int(alert_threshold)
