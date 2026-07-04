"""
conftest.py — FluidKit/tests 共通セットアップ。

FluidKit 内の各モジュールは相対 import ではなく素の `sys.path` 操作で
互いを解決している(例: gen_sample.py が `solver/` を sys.path に追加する)。
テストからも同じ流儀で FluidKit 配下のパッケージ/スクリプトを import できるよう、
FluidKit ルート・tools/・solver/ を sys.path に追加する。

本ファイルは import パス解決のみを行い、solver/tools 側のコードは一切変更しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

_FLUIDKIT_ROOT = Path(__file__).parent.parent
for _p in (_FLUIDKIT_ROOT, _FLUIDKIT_ROOT / "tools", _FLUIDKIT_ROOT / "solver"):
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
