from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from ..config import settings
from ..utils.logging_setup import get_logger
from ..pipelines.window_buffer import SlidingWindowBuffer
from ..pipelines.log_parser import parse_raw_log
from ..models.logbert_wrapper import LogBERTModel
from ..pipelines.detector import detect_anomalies, should_alert


logger = get_logger("rt-runner")


def _iter_stdin() -> Iterable[str]:
    """Yield lines from STDIN, stripping trailing newlines."""
    for line in sys.stdin:
        yield line.rstrip("\n")


def _iter_file(path: Path) -> Iterable[str]:
    """Yield lines from a file if it exists; otherwise log a warning."""
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")
    else:
        logger.warning("Log file not found: %s; no input produced", str(path))
        return


def run() -> int:
    # 1) Load config (already imported as settings)
    cfg = settings
    logger.info(
        "Starting runner: window=%d, threshold=%.3f, alert_min=%d, source=%s",
        cfg.WINDOW_SIZE,
        cfg.THRESHOLD,
        cfg.ALERT_ANOMALY_COUNT,
        cfg.STREAM_SOURCE,
    )

    # 2) Initialize components
    window = SlidingWindowBuffer(window_size=cfg.WINDOW_SIZE)
    # Prefer real mode if external model paths are provided; else default to mock
    use_real = bool(cfg.LOGBERT_MODEL_PATH and cfg.LOGBERT_VOCAB_PATH)
    if use_real:
        logger.info(
            "Initializing LogBERT real mode: model=%s vocab=%s device=%s",
            cfg.LOGBERT_MODEL_PATH,
            cfg.LOGBERT_VOCAB_PATH,
            cfg.LOGBERT_DEVICE or "auto",
        )
        model = LogBERTModel(
            mode="real",
            model_path=cfg.LOGBERT_MODEL_PATH,
            vocab_path=cfg.LOGBERT_VOCAB_PATH,
            device=cfg.LOGBERT_DEVICE,
        )
    else:
        logger.info("Initializing LogBERT mock mode (no external checkpoint configured)")
        model = LogBERTModel(mode="mock")

    # 3) Select input stream
    if (cfg.STREAM_SOURCE or "file").lower() == "stdin":
        it = _iter_stdin()
        logger.info("Reading input from STDIN ...")
    else:
        path = Path(cfg.LOG_FILE_PATH)
        it = _iter_file(path)
        logger.info("Reading input from file: %s", str(path))

    processed = 0
    for raw in it:
        processed += 1
        key = parse_raw_log(raw)
        keys = window.add(key)

        if window.size() >= cfg.WINDOW_SIZE:
            # 4) Perform detection on current window
            probs = model.predict_probabilities(keys)
            anomalies = detect_anomalies(keys, probs, cfg.THRESHOLD)
            do_alert = should_alert(len(anomalies), cfg.ALERT_ANOMALY_COUNT)

            if anomalies:
                # Keep concise list of indices with probs
                preview = ", ".join(f"{idx}:{p:.3f}" for idx, _k, p in anomalies[:5])
                msg = (
                    f"processed={processed} window={len(keys)} anomalies={len(anomalies)} "
                    f"thr={cfg.THRESHOLD:.3f} [{preview}]"
                )
                if do_alert:
                    logger.warning("ALERT %s", msg)
                else:
                    logger.info("%s", msg)
            else:
                logger.info(
                    "processed=%d window=%d anomalies=0 thr=%.3f",
                    processed,
                    len(keys),
                    cfg.THRESHOLD,
                )
        else:
            logger.info("warming-up processed=%d window=%d/%d", processed, window.size(), cfg.WINDOW_SIZE)

    logger.info("Input exhausted. Total processed: %d", processed)
    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
