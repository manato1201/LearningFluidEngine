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


class FluidDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.Y = torch.tensor(np.load(Y_path), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train_v2] デバイス: {device}")

    # データ
    tr_ds = FluidDataset(ds/"X_train.npy", ds/"Y_train.npy")
    va_ds = FluidDataset(ds/"X_val.npy",   ds/"Y_val.npy")
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
        for X, Y in tr_dl:
            X, Y = X.to(device), Y.to(device)
            with autocast():
                pred     = model.predict_next(X)
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
            for X, Y in va_dl:
                X, Y = X.to(device), Y.to(device)
                with autocast():
                    pred    = model.predict_next(X)
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
            mark = " ← best"

        if ep % 10 == 0 or ep == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Epoch {ep:4d}/{args.epochs}  "
                  f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={lr_now:.6f}  ({time.time()-t0:.1f}s){mark}")

    torch.save({"model_state": model.state_dict(), "config": vars(args)},
               ckpt_dir / "last.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train_v2] 完了  best_val={best_val:.5f}")
    print(f"  チェックポイント: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
