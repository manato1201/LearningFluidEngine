"""
train_v2.py  —  FluidKit Neural Fluid v2 訓練スクリプト

改善点:
  - dataset_v2 (100シム) を使用
  - NeuralFluidV2 (latent=128, mp_steps=3)
  - FluidLossV2 (位置+速度+運動量)
  - Cosine Annealing スケジューラ
  - 混合精度 (AMP) で GPU メモリ節約

使い方:
    python train_v2.py
    python train_v2.py --epochs 150 --batch 32
"""

import json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from model_v2 import NeuralFluidV2, FluidLossV2, count_params


def half_state_dict(model: nn.Module) -> dict:
    """学習中のモデル自体には触れず、CPU上に複製した state_dict のみを
    fp16化して返す（model.half() を実モデルに適用すると AMP/optimizer の
    状態と整合しなくなるため厳禁）。整数バッファ等（非浮動小数）はそのまま維持する。
    """
    return {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
            for k, v in model.state_dict().items()}


class FluidDataset(Dataset):
    """X: (T,N,6) pos+vel, Y: (T,N,3) next pos, NBR: (T,N,k) 事前計算済み近傍インデックス（cKDTree由来）

    データセット全体をメモリに読み込まず np.load(..., mmap_mode='r') で
    メモリマップし、__getitem__ でアクセスされたスライスだけをオンデマンドで
    torch.tensor に変換する（torch は numpy のような mmap を直接サポートしない
    ため、__init__ で全体を torch.tensor 化すると mmap の意味がなくなる）。
    """
    def __init__(self, X_path, Y_path, NBR_path):
        self.X   = np.load(X_path,   mmap_mode='r')
        self.Y   = np.load(Y_path,   mmap_mode='r')
        self.NBR = np.load(NBR_path, mmap_mode='r')

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x   = torch.tensor(np.asarray(self.X[i]),   dtype=torch.float32)
        y   = torch.tensor(np.asarray(self.Y[i]),   dtype=torch.float32)
        nbr = torch.tensor(np.asarray(self.NBR[i]), dtype=torch.long)
        return x, y, nbr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default=str(Path(__file__).parent / "dataset_v2"))
    ap.add_argument("--ckpt",     default=str(Path(__file__).parent / "checkpoints_v2"))
    ap.add_argument("--epochs",   type=int,   default=120)
    ap.add_argument("--batch",    type=int,   default=32)
    ap.add_argument("--lr",       type=float, default=2e-3)
    ap.add_argument("--latent",   type=int,   default=128)
    ap.add_argument("--hidden",   type=int,   default=256)
    ap.add_argument("--k",        type=int,   default=16)
    ap.add_argument("--mp",       type=int,   default=3)
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not (ds / "X_train.npy").exists():
        print(f"[ERROR] データセット未作成: {ds}")
        print("  先に collect_data.py --simulations 100 --output ./dataset_v2 を実行")
        return
    if not (ds / "NBR_train.npy").exists():
        print(f"[ERROR] 近傍インデックス未作成: {ds}/NBR_train.npy")
        print("  collect_data.py を最新版で再実行してください（cKDTree事前計算に対応）")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train_v2] デバイス: {device}")

    # データ
    tr_ds = FluidDataset(ds/"X_train.npy", ds/"Y_train.npy", ds/"NBR_train.npy")
    va_ds = FluidDataset(ds/"X_val.npy",   ds/"Y_val.npy",   ds/"NBR_val.npy")
    tr_dl = DataLoader(tr_ds, args.batch, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # モデル
    model   = NeuralFluidV2(latent=args.latent, hidden=args.hidden,
                             k_neighbors=args.k, n_mp_steps=args.mp).to(device)
    loss_fn = FluidLossV2()
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler  = GradScaler()

    print(f"  パラメータ数: {count_params(model):,}")
    print(f"  訓練: {len(tr_ds)} / 検証: {len(va_ds)} サンプル")
    print(f"  エポック: {args.epochs}  バッチ: {args.batch}  lr: {args.lr}\n")

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history  = {"train": [], "val": []}

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        # 訓練
        model.train()
        tr_loss = 0.0
        for X, Y, NBR in tr_dl:
            X, Y, NBR = X.to(device), Y.to(device), NBR.to(device)
            with autocast():
                pred     = model.predict_next(X, NBR)
                loss, _  = loss_fn(pred, X, Y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(X)
        tr_loss /= len(tr_ds)

        # 検証
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, Y, NBR in va_dl:
                X, Y, NBR = X.to(device), Y.to(device), NBR.to(device)
                with autocast():
                    pred    = model.predict_next(X, NBR)
                    loss, _ = loss_fn(pred, X, Y)
                va_loss += loss.item() * len(X)
        va_loss /= len(va_ds)
        sched.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        mark = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(),
                        "config": vars(args), "best_val": best_val},
                       ckpt_dir / "best.pt")
            # 追加: fp16 版チェックポイント（既存 fp32 の best.pt は変更せず維持）
            torch.save({"model_state": half_state_dict(model),
                        "config": vars(args), "best_val": best_val, "fp16": True},
                       ckpt_dir / "best_fp16.pt")
            mark = " ← best"

        if ep % 10 == 0 or ep == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Epoch {ep:4d}/{args.epochs}  "
                  f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={lr_now:.6f}  ({time.time()-t0:.1f}s){mark}")

    torch.save({"model_state": model.state_dict(), "config": vars(args)},
               ckpt_dir / "last.pt")
    # 追加: fp16 版チェックポイント（既存 fp32 の last.pt は変更せず維持）
    torch.save({"model_state": half_state_dict(model), "config": vars(args), "fp16": True},
               ckpt_dir / "last_fp16.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train_v2] 完了  best_val={best_val:.5f}")
    print(f"  チェックポイント: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
