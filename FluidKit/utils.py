"""
utils.py  —  FluidKit 共通ユーティリティ

Neural Fluid パイプライン（collect_data.py / infer_v2.py）で重複していた
正規化・逆正規化ロジックを集約する。

数値挙動は元の実装（collect_data.py の norm_pos/norm_vel、
infer_v2.py の get_initial_state 内インライン計算・denorm_pos）と
完全に同一（定数・計算式とも変更なし）。
"""

import numpy as np


def normalize_pos(pos, pos_min, pos_max):
    """位置を [pos_min, pos_max] → [-1, 1] へ正規化する。

    collect_data.py の norm_pos() / infer_v2.py の pos_norm 計算と同一の式。
    """
    pos_range = pos_max - pos_min
    return (pos - pos_min) / pos_range * 2 - 1


def denormalize_pos(pos_norm, pos_min, pos_max):
    """[-1, 1] → [pos_min, pos_max] へ逆正規化する（normalize_pos の逆変換）。

    infer_v2.py の denorm_pos() と同一の式。
    """
    pos_range = pos_max - pos_min
    return (pos_norm + 1) / 2 * pos_range + pos_min


def normalize_vel(vel, vel_std, n_sigma=3.0):
    """速度を ±n_sigma シグマがおおむね [-1, 1] に収まるよう正規化する。

    collect_data.py の norm_vel() / infer_v2.py の vel_norm 計算と同一の式。
    """
    return vel / (vel_std * n_sigma)


def denormalize_vel(vel_norm, vel_std, n_sigma=3.0):
    """normalize_vel の逆変換。"""
    return vel_norm * (vel_std * n_sigma)
