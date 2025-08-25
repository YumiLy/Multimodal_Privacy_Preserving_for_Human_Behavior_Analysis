#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def load_xy(emb_dir: Path, tag: str, layer="L2"):
    X = np.load(emb_dir / f"{tag}_{layer}.npy")          # [N, D]
    meta = pd.read_csv(emb_dir / f"{tag}_meta.csv")      # image, emotion_remap, gender
    g = meta["gender"].astype(str).str.lower().values
    keep = np.isin(g, ["male","female"])
    y = (g[keep] == "female").astype(int)                # male=0, female=1
    return X[keep], y

def uar(y_true, y_pred):  # macro recall (binary => chance≈0.5)
    return recall_score(y_true, y_pred, average="macro")

# ----- shallow MLP -----
class ProbeMLP(nn.Module):
    def __init__(self, in_dim, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def train_mlp(Xtr, ytr, Xva, yva, epochs=200, bs=256, lr=3e-4, patience=10, device="cuda"):
    sc = StandardScaler().fit(Xtr)
    Xtr = torch.from_numpy(sc.transform(Xtr).astype(np.float32))
    Xva = torch.from_numpy(sc.transform(Xva).astype(np.float32))
    ytr = torch.from_numpy(ytr.astype(np.int64))
    yva = torch.from_numpy(yva.astype(np.int64))

    tr = DataLoader(TensorDataset(Xtr,ytr), batch_size=bs, shuffle=True)
    va = DataLoader(TensorDataset(Xva,yva), batch_size=bs, shuffle=False)

    model = ProbeMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit= nn.CrossEntropyLoss()

    best, bad = None, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # val UAR 早停
        model.eval()
        with torch.no_grad():
            pv, yv = [], []
            for xb, yb in va:
                pv.append(model(xb.to(device)).argmax(1).cpu().numpy())
                yv.append(yb.numpy())
        pv = np.concatenate(pv); yv = np.concatenate(yv)
        u = uar(yv, pv)

        if best is None or u > best[0]:
            best = (u, model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    model.load_state_dict(best[1])
    return model, sc

def eval_probe(model, sc, X, y, device="cuda"):
    X = torch.from_numpy(sc.transform(X).astype(np.float32))
    with torch.no_grad():
        p = model(X.to(device)).argmax(1).cpu().numpy()
    return accuracy_score(y, p), uar(y, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain_dir", type=Path, required=True, help="baseline embeddings dir (…/embeddings)")
    ap.add_argument("--priv_dir",  type=Path, required=True, help="GRL embeddings dir (…/embeddings)")
    ap.add_argument("--case", choices=["baseline","ignorant","informed_model", "informed_all"], required=True)
    ap.add_argument("--layer", default="L2")
    ap.add_argument("--probe", choices=["logreg","mlp"], default="logreg")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--plain_train_tag", default="attacker_train")
    ap.add_argument("--plain_val_tag",   default="attacker_val")
    ap.add_argument("--priv_train_tag",  default="attacker_train",
                    help="informed 默认用 attacker_* 训练；要跑 informed(model+data) 时设为 main_train")
    ap.add_argument("--priv_val_tag",    default="attacker_val",
                    help="informed 默认用 attacker_* 验证；要跑 informed(model+data) 时设为 main_val")

    args = ap.parse_args()

    print(f"=== Attacker probe: case={args.case} layer={args.layer} probe={args.probe} ===")
    # 1) 读数据
    if args.case == "baseline":
        Xtr,ytr = load_xy(args.plain_dir, "attacker_train", args.layer)
        Xva,yva = load_xy(args.plain_dir, "attacker_val",   args.layer)
        Xte,yte = load_xy(args.plain_dir, "test",           args.layer)
    elif args.case == "ignorant":
        Xtr,ytr = load_xy(args.plain_dir, "attacker_train", args.layer)
        Xva,yva = load_xy(args.plain_dir, "attacker_val",   args.layer)
        Xte,yte = load_xy(args.priv_dir,  "test",           args.layer)
    elif args.case == "informed_model":
        Xtr,ytr = load_xy(args.priv_dir,  "attacker_train",     args.layer)
        Xva,yva = load_xy(args.priv_dir,  "attacker_val",       args.layer)
        Xte,yte = load_xy(args.priv_dir,  "test",           args.layer)
    elif args.case == "informed_all":
        Xtr,ytr = load_xy(args.priv_dir,  "main_train", args.layer)
        Xva,yva = load_xy(args.priv_dir,  "main_val",   args.layer)
        Xte,yte = load_xy(args.priv_dir,  "test",           args.layer)

    # 2) 训练探针
    if args.probe == "logreg":
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        best = None
        for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
            clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            pv = clf.predict(Xva_s)
            u = uar(yva, pv)
            if best is None or u > best[0]:
                best = (u, C, clf, sc)
        clf, sc = best[2], best[3]
        acc, u = accuracy_score(yte, clf.predict(sc.transform(Xte))), uar(yte, clf.predict(sc.transform(Xte)))
        print(f"[{args.case}|logreg] TEST acc={acc:.3f} uar={u:.3f} (best C={best[1]})")
    else:
        model, sc = train_mlp(Xtr, ytr, Xva, yva, device=args.device)
        acc, u = eval_probe(model, sc, Xte, yte, device=args.device)
        print(f"[{args.case}|mlp]    TEST acc={acc:.3f} uar={u:.3f}")

if __name__ == "__main__":
    main()
