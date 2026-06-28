"""
reproducibility.py — シミュレーション再現性保証 + 監査ログ

Judge0 パターンの転用:
  - 実行設定を SHA-256 でハッシュ化
  - 各ステップの状態ハッシュを記録
  - numpy バージョン・シードを固定して再現可能にする
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class SimulationRecord:
    """
    シミュレーション実行の完全な再現性を保証する記録クラス。

    使い方:
        record = SimulationRecord(config, seed=42)
        for step in range(n_steps):
            solver.step()
            record.log_step(step, solver.get_state())
        record.save("audit.json")
        record.save_npz("state_history.npz")  # 状態配列ごと保存
    """

    def __init__(self, config: dict[str, Any], seed: int = 42):
        np.random.seed(seed)

        self.config = config
        self.seed   = seed
        self._steps: list[dict] = []
        self._states: list[np.ndarray] = []
        self._start_ns = time.perf_counter_ns()

        self.record: dict[str, Any] = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "config_hash":  self._hash_dict(config),
            "numpy_version": np.__version__,
            "random_seed":  seed,
            "steps":        self._steps,
        }

    # ──────────────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        state: np.ndarray,
        *,
        extra: dict | None = None,
    ) -> None:
        """ステップごとに状態ハッシュを記録。"""
        state_arr = np.asarray(state, dtype=np.float64)
        entry: dict[str, Any] = {
            "step":       step,
            "state_hash": self._hash_array(state_arr),
            "elapsed_ms": (time.perf_counter_ns() - self._start_ns) / 1e6,
        }
        if extra:
            entry.update(extra)
        self._steps.append(entry)
        self._states.append(state_arr.copy())

    def verify(self, other: "SimulationRecord") -> bool:
        """2 つの実行結果が完全に一致するか確認。"""
        if self.record["config_hash"] != other.record["config_hash"]:
            return False
        for s1, s2 in zip(self._steps, other._steps):
            if s1["state_hash"] != s2["state_hash"]:
                return False
        return True

    def save(self, path: str | Path) -> None:
        """監査ログを JSON に保存。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.record, f, indent=2, ensure_ascii=False)

    def save_npz(self, path: str | Path) -> None:
        """状態配列ごと NPZ に保存（再現 / デバッグ用）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"step_{i:05d}": arr for i, arr in enumerate(self._states)}
        arrays["config_hash"] = np.array([self.record["config_hash"]])
        np.savez_compressed(p, **arrays)

    @classmethod
    def load_and_verify(cls, path: str | Path) -> dict:
        """保存済み監査ログを読み込んで内容を返す。"""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ──────────────────────────────────────────────────────

    @staticmethod
    def _hash_dict(d: dict) -> str:
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()

    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()
