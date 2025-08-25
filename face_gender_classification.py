#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face (RAF-DB) gender-only classifier on pooled CNN features.
Splits: main_train / main_val / test
Files:  <split_dir>/{main_train,main_val,test}_feats.npy
        <split_dir>/{main_train,main_val,test}_labels.csv  (columns: path, emotion, gender)
"""

import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_face_split(split_dir: Path, tag: str):
    X = np.load(split_dir / f"{tag}_feats.npy")                  # (N, D=2048)
    df = pd.read_csv(split_dir / f"{tag}_labels.csv")            # columns: path, emotion, gender
    g = df["gender"].astype(str).str.lower().values
    keep = np.isin(g, ["male", "female"])                        # 仅保留有性别的样本
    y = (g[keep] == "female").astype(np.int64)                   # male=0, female=1
    return X[keep].astype(np.float32), y

class GenderMLP(nn.Module):
    def __init__(self, in_dim=2048, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(drop),
            nn.Linear(512, 128),    nn.ReLU(inplace=True), nn.BatchNorm1d(128), nn.Dropout(drop),
            nn.Linear(128, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def uar(y_true, y_pred):  # macro-recall (binary: chance ~0.5)
    return recall_score(y_true, y_pred, average="macro")

def run_epoch(model, loader, crit=None, opt=None, device="cuda"):
    train = opt is not None
    model.train() if train else model.eval()
    losses, yt, yp = [], [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if train:
                loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item()*xb.size(0))
            yt.append(yb.cpu().numpy())
            yp.append(logits.argmax(1).detach().cpu().numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    acc = accuracy_score(yt, yp); uu = uar(yt, yp)
    loss = (sum(losses)/len(loader.dataset)) if train else None
    return loss, acc, uu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=Path, required=True,
                    help="Dir with *_feats.npy and *_labels.csv (RAF-DB pooled features)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs",     type=int, default=256)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", choices=["cuda","cpu","auto"], default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)
    print(f"Device: {device}")

    # 1) Load splits (filter unknown gender)
    Xtr, ytr = load_face_split(args.split_dir, "main_train")
    Xva, yva = load_face_split(args.split_dir, "main_val")
    Xte, yte = load_face_split(args.split_dir, "test")
    in_dim = Xtr.shape[1]
    print(f"Shapes  train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # 2) Standardize (fit on train only)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    dump(scaler, args.split_dir / "gender_only_scaler.joblib")

    # 3) DataLoaders
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=args.bs, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=args.bs, shuffle=False)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                    batch_size=args.bs, shuffle=False)

    # 4) Model / train with early stop on val UAR
    model = GenderMLP(in_dim=in_dim, drop=0.3).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = (-1.0, None); bad = 0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_uar = run_epoch(model, tr, crit, opt, device)
        _, va_acc, va_uar = run_epoch(model, va, None, None, device)
        print(f"Ep{ep:03d} | tr L {tr_loss:.4f} acc {tr_acc:.3f} uar {tr_uar:.3f} "
              f"| va acc {va_acc:.3f} uar {va_uar:.3f}")
        if va_uar > best[0]:
            best = (va_uar, {k: v.detach().cpu() for k,v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep}. Best val UAR={best[0]:.4f}")
                break

    # 5) Test with the best checkpoint
    model.load_state_dict(best[1], strict=True)
    _, te_acc, te_uar = run_epoch(model, te, None, None, device)
    print(f"[TEST] acc={te_acc:.4f}  UAR={te_uar:.4f}")

if __name__ == "__main__":
    main()
