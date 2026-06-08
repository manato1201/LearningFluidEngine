"""
train.py  —  FluidKit Neural Fluid 訓練スクリプト

使い方:
    # データ収集から訓練まで一括
    python collect_data.py --simulations 20
    python train.py

    # オプション指定
    python train.py --epochs 100 --batch 32 --lr 1e-3 --dataset ./dataset

    # GPU 使用（自動検出）
    python train.py --epochs 200

訓練後の推論:
    python infer_to_json.py --checkpoint ./checkpoints/best.pt
"""

import json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralFluidModel, FluidLoss, count_params


# ──────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────

class FluidDataset(Dataset):
    """
    X: (n_samples, n_particles, 6)  [pos + vel] at t
    Y: (n_samples, n_particles, 3)  [pos] at t+1
    """
    def __init__(self, X_path: Path, Y_path: Path):
        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.Y = torch.tensor(np.load(Y_path), dtype=torch.float32)
        assert len(self.X) == len(self.Y), "X/Y サイズ不一致"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, ckpt_dir: Path):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.device    = device
        self.ckpt_dir  = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val  = float("inf")
        self.history   = {"train": [], "val": []}

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)
                pred_pos = self.model.predict_next(X)       # (B,N,3)
                loss, _ = self.loss_fn(pred_pos, Y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += loss.item() * len(X)
        return total_loss / len(loader.dataset)

    def fit(self, train_loader, val_loader, epochs: int, scheduler=None):
        print(f"\n{'─'*60}")
        print(f"  デバイス: {self.device}")
        print(f"  パラメータ数: {count_params(self.model):,}")
        print(f"  訓練サンプル: {len(train_loader.dataset)}")
        print(f"  検証サンプル: {len(val_loader.dataset)}")
        print(f"{'─'*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr  = self._run_epoch(train_loader, train=True)
            val = self._run_epoch(val_loader,   train=False)
            dt  = time.time() - t0

            self.history["train"].append(tr)
            self.history["val"].append(val)

            if scheduler:
                scheduler.step(val)

            mark = ""
            if val < self.best_val:
                self.best_val = val
                self._save("best.pt")
                mark = "  ← best"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs}  "
                      f"train={tr:.5f}  val={val:.5f}  "
                      f"({dt:.1f}s){mark}")

        self._save("last.pt")
        self._save_history()
        print(f"\n[train] 完了  best_val={self.best_val:.5f}")

    def _save(self, name: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val": self.best_val,
            "history": self.history,
        }, self.ckpt_dir / name)

    def _save_history(self):
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default=str(Path(__file__).parent / "dataset"))
    ap.add_argument("--ckpt",       default=str(Path(__file__).parent / "checkpoints"))
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--batch",      type=int,   default=16)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--latent",     type=int,   default=64)
    ap.add_argument("--hidden",     type=int,   default=128)
    ap.add_argument("--k",          type=int,   default=12,   help="近傍粒子数")
    ap.add_argument("--mp-steps",   type=int,   default=2,    help="メッセージパッシング回数")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=0.01)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    if not (dataset_dir / "X_train.npy").exists():
        print("[ERROR] データセットが見つかりません。先に collect_data.py を実行してください。")
        print(f"        python collect_data.py --output {dataset_dir}")
        return

    # ── データ ──────────────────────────
    train_ds = FluidDataset(dataset_dir / "X_train.npy", dataset_dir / "Y_train.npy")
    val_ds   = FluidDataset(dataset_dir / "X_val.npy",   dataset_dir / "Y_val.npy")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── メタ情報 ────────────────────────
    meta_path = dataset_dir / "meta.json"
    in_dim = 6
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        in_dim = meta.get("input_dim", 6)

    # ── モデル ──────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralFluidModel(
        in_dim=in_dim, latent=args.latent,
        hidden=args.hidden, k_neighbors=args.k, n_mp_steps=args.mp_steps,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, verbose=True
    )
    loss_fn = FluidLoss(pos_weight=args.pos_weight, smooth_weight=args.smooth_weight)

    trainer = Trainer(model, optimizer, loss_fn, device, Path(args.ckpt))
    trainer.fit(train_dl, val_dl, args.epochs, scheduler)


if __name__ == "__main__":
    main()
