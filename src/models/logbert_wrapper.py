from __future__ import annotations

import math
import sys
from hashlib import sha256
from pathlib import Path
from typing import List, Optional


class LogBERTModel:
    """LogBERT wrapper with mock and real (external/logbert) modes.

    - mode="mock": deterministic pseudo-probabilities for tests and demos.
    - mode="real": load external/logbert checkpoint and run masked-LM scoring in-process.
    """

    def __init__(
        self,
        mode: str = "real",
        *,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        mode = (mode or "mock").lower()
        if mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        self.mode = mode

        self._torch = None
        self._model = None
        self._vocab = None
        self._device = "cpu"

        if self.mode == "real":
            self._setup_external_logbert()
            try:
                import torch  # noqa: F401
                from bert_pytorch.dataset import WordVocab  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Failed to import external/logbert modules. Ensure external/logbert is present and torch is installed."
                ) from e
            self._torch = __import__("torch")

            model_path = model_path or str(
                Path(__file__).resolve().parents[2] / "external" / "logbert" / "output" / "hdfs" / "bert" / "best_bert.pth"
            )
            vocab_path = vocab_path or str(
                Path(__file__).resolve().parents[2] / "external" / "logbert" / "output" / "hdfs" / "vocab.pkl"
            )

            self._device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")

            # Load model (torch.save(self.model) format) and vocab
            self._model = self._torch.load(model_path, map_location=self._device)
            self._model.to(self._device)
            self._model.eval()
            self._vocab = WordVocab.load_vocab(vocab_path)

    def predict_probabilities(self, sequence_keys: List[str]) -> List[float]:
        """Return per-event probabilities for a sequence of log keys.

        - Mock: deterministic values in [0.02, 0.99].
        - Real: masked-LM scoring with external/logbert model (one mask per position).
        """
        if self.mode == "mock":
            return [self._mock_probability_from_key(k) for k in sequence_keys]
        return self._predict_real(sequence_keys)

    # ---------------- Mock helpers ----------------
    @staticmethod
    def _mock_probability_from_key(key: str) -> float:
        if not isinstance(key, str):
            key = str(key)
        digest = sha256(key.encode("utf-8")).digest()
        n = int.from_bytes(digest, byteorder="big", signed=False)
        denom = (1 << 256) - 1
        r = n / denom
        lo, hi = 0.02, 0.99
        return lo + r * (hi - lo)

    # ---------------- Real mode helpers ----------------
    def _setup_external_logbert(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ext = root / "external" / "logbert"
        if str(ext) not in sys.path:
            sys.path.insert(0, str(ext))

    def _predict_real(self, keys: List[str]) -> List[float]:
        if self._model is None or self._vocab is None or self._torch is None:
            raise RuntimeError("Real model not initialized. Instantiate with mode='real' and valid paths.")

        # Convert keys to vocab indices; prepend SOS, build time zeros
        ids = [self._vocab.sos_index] + [self._vocab.stoi.get(k, self._vocab.unk_index) for k in keys]
        L = len(ids)
        time_input = [0.0] * L

        probs: List[float] = []
        for pos in range(1, L):  # skip SOS at 0
            masked_ids = ids.copy()
            true_id = masked_ids[pos]
            masked_ids[pos] = self._vocab.mask_index

            bert_input = self._torch.tensor([masked_ids], dtype=self._torch.long, device=self._device)
            time_tensor = self._torch.tensor([[t] for t in time_input], dtype=self._torch.float, device=self._device)
            time_tensor = time_tensor.unsqueeze(0)  # (1, L, 1)

            with self._torch.inference_mode():
                out = self._model.forward(bert_input, time_tensor)
                log_probs = out["logkey_output"]  # (1, L, V) in log-softmax
                lp = float(log_probs[0, pos, true_id].item())
                probs.append(math.exp(lp))

        return probs
