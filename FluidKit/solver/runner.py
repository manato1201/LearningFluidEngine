"""
runner.py — タイムアウト付きサンドボックス実行ラッパー

Judge0 パターンの転用:
  - SIGALRM (Unix) / threading.Timer (Windows) によるタイムアウト
  - 設定バリデーション + 再現性記録を自動付与
  - 戻り値は { status, frames, hash, elapsed_ms, error? }
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

from .reproducibility import SimulationRecord


# ──────────────────────────────────────────────────────────
#  タイムアウト (Unix: SIGALRM / Windows: Timer スレッド)
# ──────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int) -> Generator[None, None, None]:
    """
    クロスプラットフォームなタイムアウトコンテキストマネージャ。

    Unix:    signal.SIGALRM (メインスレッドのみ有効)
    Windows: daemon Timer スレッドで _TimeoutError を raise
    """
    if sys.platform != "win32":
        import signal

        def _handler(signum: int, frame: Any) -> None:
            raise _TimeoutError(f"Simulation exceeded {seconds}s limit")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        _flag: dict[str, bool] = {"expired": False}
        _main_thread = threading.current_thread()

        def _fire() -> None:
            _flag["expired"] = True
            # Windows では直接例外を投げられないため、
            # メインスレッドに KeyboardInterrupt を送る
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(_main_thread.ident),      # type: ignore[arg-type]
                ctypes.py_object(_TimeoutError),
            )

        timer = threading.Timer(seconds, _fire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()


# ──────────────────────────────────────────────────────────
#  設定バリデーション
# ──────────────────────────────────────────────────────────

_CONFIG_SCHEMA: dict[str, tuple[type, Any, Any]] = {
    # key: (type, min, max)  — min/max=None はチェックなし
    "N":          (int,   8,    512),
    "dt":         (float, 1e-5, 1.0),
    "visc":       (float, 0.0,  10.0),
    "n_frames":   (int,   1,    2000),
    "seed":       (int,   0,    None),
}


def validate_config(config: dict) -> dict:
    """
    設定辞書を検証し、不正な値はデフォルト値で置換。
    Returns: 検証済み設定 (コピー)。
    """
    defaults: dict[str, Any] = {
        "N": 64, "dt": 0.05, "visc": 0.0, "n_frames": 100, "seed": 42,
    }
    out = {**defaults, **config}

    for key, (typ, lo, hi) in _CONFIG_SCHEMA.items():
        val = out.get(key, defaults.get(key))
        try:
            val = typ(val)
        except (TypeError, ValueError):
            val = defaults[key]
        if lo is not None and val < lo:
            val = lo
        if hi is not None and val > hi:
            val = hi
        out[key] = val

    return out


# ──────────────────────────────────────────────────────────
#  サンドボックス実行エントリポイント
# ──────────────────────────────────────────────────────────

def run_simulation_sandboxed(
    config: dict,
    solver_factory,         # Callable[[dict], solver] — ソルバー生成関数
    timeout_sec: int = 300,
    audit_path: str | None = None,
) -> dict:
    """
    Judge0 パターン: バリデーション + タイムアウト + 再現性記録付きで実行。

    Args:
        config:          シミュレーション設定
        solver_factory:  config を受け取りソルバーを返す関数
        timeout_sec:     最大実行時間 [秒]
        audit_path:      監査ログの保存先 (None なら保存しない)

    Returns:
        {
            "status":     "completed" | "timeout" | "error",
            "frames":     int,
            "state_hash": str,     # 最終フレームのハッシュ
            "elapsed_ms": float,
            "error":      str,     # エラー時のみ
        }
    """
    cfg = validate_config(config)
    record = SimulationRecord(cfg, seed=cfg["seed"])
    t0 = time.perf_counter()

    try:
        with timeout(timeout_sec):
            solver = solver_factory(cfg)
            n_frames = cfg["n_frames"]

            for step in range(n_frames):
                state = solver.step()
                record.log_step(step, state)

    except _TimeoutError:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "status":     "timeout",
            "frames":     len(record._steps),
            "state_hash": "",
            "elapsed_ms": elapsed,
            "error":      f"Exceeded {timeout_sec}s limit",
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "status":     "error",
            "frames":     len(record._steps),
            "state_hash": "",
            "elapsed_ms": elapsed,
            "error":      str(exc),
        }

    elapsed = (time.perf_counter() - t0) * 1000

    if audit_path:
        record.save(audit_path)

    last_hash = record._steps[-1]["state_hash"] if record._steps else ""
    return {
        "status":     "completed",
        "frames":     n_frames,
        "state_hash": last_hash,
        "elapsed_ms": elapsed,
    }
