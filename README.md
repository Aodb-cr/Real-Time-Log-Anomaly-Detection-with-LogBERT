# Real-Time Log Anomaly Detection with LogBERT

Lightweight, real-time log anomaly detection using a sliding window buffer and a (Log)BERT masked language model, plus a Streamlit dashboard for monitoring and alerting.

## Overview

- Sliding window collects recent logs efficiently (`deque`-based, count/time window).
- Parser normalizes dynamic parts into templates (IPs, numbers, paths, etc.).
- LogBERT (Masked LM) scores likelihood per event via masked prediction.
- Detector flags low-probability events with configurable thresholds/top-g.
- Dashboard visualizes logs, anomalies, metrics, alerts, and history.

## Architecture Flow

1) Ingest logs → 2) Window Buffer → 3) Parser/Template → 4) LogBERT Wrapper (MLM scoring) → 5) Detector (threshold/top-g, severity, alerting) → 6) Dashboard/Exporter

## Quick Start

1) Create and activate a virtual environment
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `py -m venv .venv; .\.venv\Scripts\Activate.ps1`
2) Install dependencies: `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and adjust values (e.g., `LOGBERT_MODEL`).

## How to Run

1) Simulate logs (writes sample logs to stdout or a pipe):
   - `python -m ..runners.stream_simulator`
2) Run pipeline (ingest → detect → print alerts):
   - `python -m ..runners.main`
3) Launch dashboard (real-time monitoring):
   - `streamlit run src/dashboards/streamlit_app.py`

## Swap Mock to Real LogBERT

- Model wrapper: `src/models/logbert_wrapper.py`
  - Load from HuggingFace Hub: set env `LOGBERT_MODEL=your-org/your-logbert` or pass `hf_model` in code.
  - Load local checkpoint (HuggingFace format with `config.json` + weights): set `LOGBERT_LOCAL_PATH=/path/to/dir`.
  - Device: `LOGBERT_DEVICE=cpu|cuda` (or leave unset for auto).
- Requirements: ensure `torch` (matching your CUDA/CPU build) and `transformers` are installed. The pinned versions in `requirements.txt` are CPU-friendly; for CUDA, follow PyTorch install docs.
- Dashboard: use the sidebar to select model source (Hub/local) and device, then “Load / Reload model”.
- Pipeline: the detector will use the wrapper’s `score_sequence` for probabilities; once a real LogBERT is loaded, it replaces the mock/heuristic scoring automatically.

## Structure

```
logbert-rt-anomaly/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ src/
│  ├─ config.py
│  ├─ pipelines/
│  │  ├─ window_buffer.py
│  │  ├─ log_parser.py
│  │  └─ detector.py
│  ├─ models/
│  │  └─ logbert_wrapper.py
│  ├─ dashboards/
│  │  └─ streamlit_app.py
│  ├─ runners/
│  │  ├─ stream_simulator.py
│  │  └─ main.py
│  └─ utils/
│     └─ logging_setup.py
├─ data/
│  └─ sample_logs.txt
└─ tests/
   ├─ test_window_buffer.py
   └─ test_detector.py
```
