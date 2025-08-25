#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attacker on layer embeddings (gender)
------------------------------------
- Baseline / Informed: use --exp
- Ignorant:            use --exp-train and --exp-test
Each exp dir must contain per-fold subdirs with `embeddings/`:
  embeddings/{trans_train,trans_val,test}_{L1,L2,L3,Final}.npy
  embeddings/{trans_train,trans_val,test}_meta.csv  (has column 'gender')

Examples
  # Baseline (plain->plain) or Informed (priv->priv)
  python attacker_probe.py --exp /path/to/exp/layer4_lam0.6 --layer L4 --model logreg

  # Ignorant (train on plain, test on priv)
  python attacker_probe.py \
     --exp-train /path/to/exp_plain \
     --exp-test  /path/to/exp_priv_layer4_lam0.6 \
     --layer L4 --model mlp
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import csv

# ---------- I/O helpers ----------
def list_folds(root: Path):
    """Return sorted fold names by trailing integer (fold_1 ... fold_5)."""
    folds = [p.name for p in root.iterdir() if p.is_dir()]
    folds = [f for f in folds if (root/f/"embeddings").exists()]
    return sorted(folds, key=lambda x: int(x.split("_")[-1]))

def _layer_to_file(layer: str) -> str:
    # your dumps use L1/L2/L3/Final; map L4->Final for convenience
    return "Final" if layer.upper()=="L4" else layer.upper()

def load_split_emb(exp_fold_dir: Path, tag: str, layer: str):
    lay = _layer_to_file(layer)
    emb = np.load(exp_fold_dir / "embeddings" / f"{tag}_{lay}.npy")
    meta= pd.read_csv(exp_fold_dir / "embeddings" / f"{tag}_meta.csv")
    g = meta["gender"].astype(str).str.lower().map({"male":0,"female":1}).values
    keep = (g==0) | (g==1)
    return emb[keep], g[keep]

def eval_acc_uar(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro")
    return acc, uar

# ---------- tiny MLP attacker ----------
class MLP1(nn.Module):
    def __init__(self, in_dim, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def train_torch_mlp(Xtr, ytr, Xva, yva, seed=42, max_epochs=200, bs=64, lr=1e-3, patience=10, device="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
                    batch_size=bs, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva).long()),
                    batch_size=bs, shuffle=False)

    model = MLP1(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit= nn.CrossEntropyLoss()

    best = (0.0, None); bad=0
    for ep in range(1, max_epochs+1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()

        # validate (UAR early-stop)
        model.eval()
        with torch.no_grad():
            xs = torch.from_numpy(Xva).to(device)
            yp = model(xs).argmax(1).cpu().numpy()
            uar = recall_score(yva, yp, average="macro")
        if uar > best[0]:
            best = (uar, {k:v.detach().cpu() for k,v in model.state_dict().items()}); bad=0
        else:
            bad += 1
            if bad >= patience: break

    model.load_state_dict(best[1])
    return model, scaler

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", type=Path, help="single exp dir (Baseline / Informed)")
    group.add_argument("--exp_train", type=Path, help="Ignorant: train/val exp dir (plain)")
    ap.add_argument("--exp_test",  type=Path, help="Ignorant: test exp dir (priv)")
    ap.add_argument("--layer", choices=["L1","L2","L3","L4"], default="L2")
    ap.add_argument("--model", choices=["logreg","mlp"], default="logreg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_prefix", type=str, default="trans",
                help="Prefix for training embeddings (e.g., trans/risk)")
    ap.add_argument("--val_prefix", type=str, default="trans",
                help="Prefix for validation embeddings (e.g., trans/risk)")
    ap.add_argument("--test_tag", type=str, default="test",
                help="Base tag for test embeddings (usually 'test' or 'main_test')")
    ap.add_argument("--out_csv", type=Path, help="Append results (one row) to this CSV file.")
    args = ap.parse_args()

    # decide folds & per-fold directories
    if args.exp is not None:
        # single exp (baseline / informed)
        folds = list_folds(args.exp)
        plan = [(args.exp/f, args.exp/f, f) for f in folds]  # (train_fold, test_fold, name)
        mode = "single-exp"
    else:
        assert args.exp_train is not None and args.exp_test is not None, \
            "Ignorant mode requires --exp-train and --exp-test"
        folds_tr = list_folds(args.exp_train)
        folds_te = list_folds(args.exp_test)
        assert folds_tr == folds_te, "fold names must match between train and test exps"
        plan = [(args.exp_train/f, args.exp_test/f, f) for f in folds_tr]
        mode = "ignorant"

    print(f"[mode] {mode} | layer={args.layer} | model={args.model}")
    device = "cuda" if (args.model=="mlp" and torch.cuda.is_available()) else "cpu"

    accs, uars = [], []
    for fold_train_dir, fold_test_dir, fold_name in plan:
        print(f"\n=== {fold_name} ===")
        # train/val from train_dir
        train_tag = f"{args.train_prefix}_train".replace("-", "_")
        val_tag   = f"{args.val_prefix}_val".replace("-", "_")
        test_tag  = args.test_tag  # 就是 'test' 或 'main_test' 等

        Xtr, ytr = load_split_emb(fold_train_dir, train_tag, args.layer)
        Xva, yva = load_split_emb(fold_train_dir, val_tag,   args.layer)
        Xte, yte = load_split_emb(fold_test_dir,  test_tag,  args.layer)

        if args.model == "logreg":
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr); Xva_s = scaler.transform(Xva); Xte_s = scaler.transform(Xte)
            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                                max_iter=1000, tol=1e-4, early_stopping=True,
                                validation_fraction=0.2, n_iter_no_change=10,
                                random_state=args.seed)
            clf.fit(Xtr_s, ytr)
            yp = clf.predict(Xte_s)
        else:
            model, scaler = train_torch_mlp(Xtr, ytr, Xva, yva, seed=args.seed, device=device)
            Xte_s = scaler.transform(Xte).astype(np.float32)
            with torch.no_grad():
                yp = model(torch.from_numpy(Xte_s).to(device)).argmax(1).cpu().numpy()

        acc, uar = accuracy_score(yte, yp), recall_score(yte, yp, average="macro")
        print(f"[TEST] Acc={acc:.4f}  UAR={uar:.4f}")
        accs.append(acc); uars.append(uar)

    print("\n====== 5-fold attacker summary ======")
    print(f"Acc mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"UAR mean={np.mean(uars):.4f}  std={np.std(uars):.4f}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # 尝试推断场景名
        if args.exp is not None:
            scenario = "single-exp"  # 你也可以根据传参再细分 'Baseline' vs 'Informed(model)'
        else:
            scenario = "ignorant"
        row = {
            "scenario": scenario,
            "layer": args.layer,
            "model": args.model,
            "acc_mean": float(np.mean(accs)),
            "acc_std":  float(np.std(accs)),
            "uar_mean": float(np.mean(uars)),
            "uar_std":  float(np.std(uars)),
        }
        write_header = not args.out_csv.exists()
        with args.out_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        print(f"[csv] appended -> {args.out_csv}")

if __name__ == "__main__":
    main()
