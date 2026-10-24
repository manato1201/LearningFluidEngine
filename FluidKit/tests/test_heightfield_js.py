"""
test_heightfield_js.py — viewer/js/heightfield.js (波紋の高さ場 + コースティクス) の Node テストを pytest から実行する。

heightfield.js は DOM に依存しない純粋なロジックで、`node --test` で検証する
(時間刻みを変えても発散しない・エネルギー・コースティクスの正規化など)。
Node が無い環境ではスキップする。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

JS_DIR = Path(__file__).parent.parent / "viewer" / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node が見つからない")
def test_heightfield_node_suite():
    proc = subprocess.run(
        ["node", "--test", str(JS_DIR / "heightfield.test.js")],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout[-3000:] + proc.stderr[-1000:])
