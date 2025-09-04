from __future__ import annotations

from collections import deque
from typing import Deque, List


class SlidingWindowBuffer:
    """Fixed-size sliding window buffer for log lines.

    Uses a deque with a fixed maxlen to keep the latest `window_size` log lines.
    """

    def __init__(self, window_size: int) -> None:
        """Create a new sliding window buffer.

        Parameters
        - window_size: maximum number of log lines to retain (must be > 0)
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self._dq: Deque[str] = deque(maxlen=int(window_size))

    def add(self, log_line: str) -> List[str]:
        """Append a log line and return the current window as a list."""
        self._dq.append(str(log_line))
        return list(self._dq)

    def snapshot(self) -> List[str]:
        """Return a copy of the current window as a list."""
        return list(self._dq)

    def size(self) -> int:
        """Return the number of items currently in the window."""
        return len(self._dq)
