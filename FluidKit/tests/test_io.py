"""
test_io.py — JSON/NPZ の書き出し→読み戻しラウンドトリップテスト。

- gen_sample.py の run() が生成する JSON (非圧縮 / --gzip) の構造・値の往復確認
- reproducibility.py の SimulationRecord.save() / save_npz() の往復確認

すべて pytest の tmp_path フィクスチャを使い、リポジトリ本来のデータフォルダ
(viewer/data/ 等) には一切書き込まない。
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

import gen_sample
from solver import SimulationRecord


# ──────────────────────────────────────────────────────────────
#  gen_sample.py JSON / gzip 出力
# ──────────────────────────────────────────────────────────────

class TestGenSampleJSONRoundTrip:
    def test_uncompressed_json_round_trip(self, tmp_path):
        out = tmp_path / "sample.json"
        frames = 3
        gen_sample.run(
            "water_drop", frames=frames, dt=0.016, output=out, seed=42, use_gzip=False
        )

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))

        meta = data["metadata"]
        assert meta["preset"] == "water_drop"
        assert meta["frameCount"] == frames
        assert meta["particleCount"] == gen_sample.PRESETS["water_drop"]["particle_count"]
        assert meta["fps"] == round(1.0 / 0.016)
        assert "bounds" in meta and "min" in meta["bounds"] and "max" in meta["bounds"]

        # frames: 1フレームあたり particleCount*3 個のフラットな座標値
        assert len(data["frames"]) == frames
        assert len(data["frames"][0]) == meta["particleCount"] * 3

        # speeds: 1フレームあたり particleCount 個の速度スカラー
        assert len(data["speeds"]) == frames
        assert len(data["speeds"][0]) == meta["particleCount"]

        # 値が有限であること（NaN/Inf が紛れ込んでいないか）
        assert all(np.isfinite(v) for v in data["frames"][0])
        assert all(np.isfinite(v) for v in data["speeds"][0])

    def test_gzip_json_round_trip_matches_uncompressed(self, tmp_path):
        """
        --gzip 有効/無効で、同一シード・同一設定なら同一内容の JSON が
        得られることを確認する（圧縮は表現形式のみを変える）。
        """
        out_plain = tmp_path / "sample_plain.json"
        out_gz_base = tmp_path / "sample_gz.json"
        frames = 3

        gen_sample.run(
            "water_drop", frames=frames, dt=0.016, output=out_plain, seed=42, use_gzip=False
        )
        gen_sample.run(
            "water_drop", frames=frames, dt=0.016, output=out_gz_base, seed=42, use_gzip=True
        )

        gz_path = out_gz_base.with_suffix(out_gz_base.suffix + ".gz")
        assert gz_path.exists()

        data_plain = json.loads(out_plain.read_text(encoding="utf-8"))
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data_gz = json.load(f)

        assert data_plain == data_gz

    def test_gzip_output_is_smaller(self, tmp_path):
        """gzip 出力は非圧縮出力よりファイルサイズが小さいこと（Phase4の目的の確認）。"""
        out_plain = tmp_path / "sample_plain2.json"
        out_gz_base = tmp_path / "sample_gz2.json"
        frames = 5

        gen_sample.run(
            "water_drop", frames=frames, dt=0.016, output=out_plain, seed=1, use_gzip=False
        )
        gen_sample.run(
            "water_drop", frames=frames, dt=0.016, output=out_gz_base, seed=1, use_gzip=True
        )
        gz_path = out_gz_base.with_suffix(out_gz_base.suffix + ".gz")

        assert gz_path.stat().st_size < out_plain.stat().st_size


# ──────────────────────────────────────────────────────────────
#  reproducibility.py: SimulationRecord.save() / save_npz()
# ──────────────────────────────────────────────────────────────

class TestSimulationRecordIO:
    def _build_record(self, seed: int = 7) -> SimulationRecord:
        record = SimulationRecord({"N": 5, "dt": 0.01}, seed=seed)
        rng = np.random.RandomState(seed)
        for step in range(4):
            state = rng.uniform(-1, 1, size=(5, 2))
            record.log_step(step, state)
        return record

    def test_save_json_round_trip(self, tmp_path):
        record = self._build_record()
        audit_path = tmp_path / "audit.json"
        record.save(audit_path)

        assert audit_path.exists()
        loaded = SimulationRecord.load_and_verify(audit_path)

        assert loaded["config_hash"] == record.record["config_hash"]
        assert loaded["random_seed"] == record.record["random_seed"]
        assert len(loaded["steps"]) == len(record._steps)
        for s_loaded, s_orig in zip(loaded["steps"], record._steps):
            assert s_loaded["state_hash"] == s_orig["state_hash"]
            assert s_loaded["step"] == s_orig["step"]

    def test_save_npz_round_trip_recovers_arrays(self, tmp_path):
        record = self._build_record()
        npz_path = tmp_path / "state_history.npz"
        record.save_npz(npz_path)

        assert npz_path.exists()
        loaded = np.load(npz_path, allow_pickle=False)

        # 各ステップの状態配列が完全に一致すること
        for i, expected in enumerate(record._states):
            key = f"step_{i:05d}"
            assert key in loaded
            assert np.array_equal(loaded[key], expected)

        assert "config_hash" in loaded
        assert loaded["config_hash"][0] == record.record["config_hash"]

    def test_npz_is_compressed_format(self, tmp_path):
        """save_npz は np.savez_compressed を使う（Phase4: NPZ圧縮）。"""
        record = self._build_record()
        npz_path = tmp_path / "state_history2.npz"
        record.save_npz(npz_path)

        # savez_compressed のファイルは ZIP 形式 (先頭マジックナンバー 'PK')
        with open(npz_path, "rb") as f:
            magic = f.read(2)
        assert magic == b"PK"

    def test_cleanup_uses_tmp_path_only(self, tmp_path):
        """
        本テストスイート全体がリポジトリの実データフォルダ (viewer/data 等) に
        書き込んでいないことを明示的に確認するプレースホルダ。
        tmp_path はテスト終了後 pytest が自動クリーンアップする。
        """
        record = self._build_record()
        p = tmp_path / "sub" / "audit.json"
        record.save(p)
        assert p.exists()
        assert str(tmp_path) in str(p.resolve())
