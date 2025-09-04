from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_PATH = Path("data/sample_logs.txt")
DEFAULT_DELAY = 0.1  # seconds


def _iter_lines(path: Path) -> Iterable[str]:
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")
    else:
        # Fallback sample lines if file not present
        samples = [
            "2025-09-04 10:15:30 INFO User 12345 logged in from 10.0.0.5",
            "2025-09-04 10:15:31 WARN Failed to open /var/tmp/cache file",
            "2025-09-04 10:15:32 ERROR Connection timeout after 50000 ms",
            "2025-09-04 10:15:33 INFO User 12345 requested /api/data",
        ]
        yield from samples


def main(argv: list[str] | None = None) -> int:
    """Stream logs from a file to STDOUT with a small delay.

    Usage: python -m ..runners.stream_simulator [path] [delay_seconds]
    """
    if argv is None:
        argv = sys.argv[1:]
    path = DEFAULT_PATH
    delay = DEFAULT_DELAY

    if len(argv) >= 1 and argv[0]:
        path = Path(argv[0])
    if len(argv) >= 2 and argv[1]:
        try:
            delay = float(argv[1])
        except ValueError:
            pass

    for line in _iter_lines(path):
        print(line, flush=True)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
