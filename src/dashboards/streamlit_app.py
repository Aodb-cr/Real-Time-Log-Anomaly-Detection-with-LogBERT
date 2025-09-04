from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple
import sys

# Ensure project root is importable when run via `streamlit run`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import settings
from src.pipelines.window_buffer import SlidingWindowBuffer
from src.pipelines.log_parser import parse_raw_log
from src.models.logbert_wrapper import LogBERTModel
from src.pipelines.detector import detect_anomalies


st.set_page_config(page_title="Real-Time Log Anomaly Detection", layout="wide")


def _load_lines() -> List[str]:
    p = Path("data/sample_logs.txt")
    if p.exists() and p.is_file():
        try:
            return [line.rstrip("\n") for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except Exception:
            pass
    # Fallback samples
    return [
        "2025-09-04 10:15:30 INFO User 12345 logged in from 10.0.0.5",
        "2025-09-04 10:15:31 WARN Failed to open /var/tmp/cache file",
        "2025-09-04 10:15:32 ERROR Connection timeout after 50000 ms",
        "2025-09-04 10:15:33 INFO User 12345 requested /api/data",
    ]


def _init_state() -> None:
    ss = st.session_state
    if "buffer" not in ss:
        ss.buffer = SlidingWindowBuffer(window_size=settings.WINDOW_SIZE)
    if "model" not in ss:
        ss.model = LogBERTModel(mode="real")
    if "lines" not in ss:
        ss.lines = _load_lines()
    if "ptr" not in ss:
        ss.ptr = 0
    if "total_anoms" not in ss:
        ss.total_anoms = 0
    if "last_window" not in ss:
        ss.last_window = []  # type: List[str]
    if "last_probs" not in ss:
        ss.last_probs = []  # type: List[float]
    if "last_anoms" not in ss:
        ss.last_anoms = []  # type: List[Tuple[int, str, float]]


_init_state()

st.title("Real-Time Log Anomaly Detection with LogBERT (Real)")

with st.sidebar:
    st.header("Config")
    st.text(f"WINDOW_SIZE = {settings.WINDOW_SIZE}")
    st.text(f"THRESHOLD = {settings.THRESHOLD}")
    # Model controls: toggle real/mock and set paths
    use_real = st.checkbox("Use external LogBERT (real)", value=True)
    model_path = st.text_input("Model .pth", value="external/logbert/output/hdfs/bert/best_bert.pth")
    vocab_path = st.text_input("Vocab .pkl", value="external/logbert/output/hdfs/vocab.pkl")
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=1)
    if st.button("Load model"):
        try:
            if use_real:
                dev = None if device_choice == "auto" else device_choice
                st.session_state.model = LogBERTModel(mode="real", model_path=model_path, vocab_path=vocab_path, device=dev)
                st.success("Loaded real LogBERT model")
            else:
                st.session_state.model = LogBERTModel(mode="mock")
                st.info("Loaded mock model")
        except Exception as e:
            st.error(f"Model load failed: {e}")
    try:
        st.text(f"Model mode = {st.session_state.model.mode}")
    except Exception:
        pass
    steps = st.slider("Iterations per run", min_value=1, max_value=200, value=25)
    delay = st.slider("Delay per step (s)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    show_n = st.slider("Show last N items", min_value=5, max_value=max(5, settings.WINDOW_SIZE), value=min(20, settings.WINDOW_SIZE))
    st.caption("Press 'Run' to simulate streaming updates.")

col1, col2 = st.columns(2)
col1.metric("Window Size", f"{settings.WINDOW_SIZE}")
col2.metric("Total Anomalies", f"{st.session_state.total_anoms}")

placeholder = st.empty()

def _step_once():
    ss = st.session_state
    if not ss.lines:
        return
    raw = ss.lines[ss.ptr % len(ss.lines)]
    ss.ptr += 1
    key = parse_raw_log(raw)
    keys = ss.buffer.add(key)

    probs: List[float] = []
    anoms: List[Tuple[int, str, float]] = []
    if ss.buffer.size() >= settings.WINDOW_SIZE:
        probs = ss.model.predict_probabilities(keys)
        anoms = detect_anomalies(keys, probs, settings.THRESHOLD)
        ss.total_anoms += len(anoms)

    ss.last_window = keys
    ss.last_probs = probs
    ss.last_anoms = anoms


def _render_window():
    ss = st.session_state
    keys = ss.last_window
    probs = ss.last_probs
    anoms = ss.last_anoms
    idx_to_prob = {i: p for i, _k, p in anoms}

    with placeholder.container():
        st.subheader("Latest Log Keys")
        start = max(0, len(keys) - show_n)
        for i in range(start, len(keys)):
            k = keys[i]
            if i in idx_to_prob:
                st.error(f"[{i}] {k}  (p={idx_to_prob[i]:.3f})")
            else:
                st.write(f"[{i}] {k}")


if st.button("Run"):
    for _ in range(steps):
        _step_once()
        _render_window()
        time.sleep(delay)

# initial render
_render_window()
