#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech-style Multi-task (Emotion + Gender-GRL) with 4-layer MLP
----------------------------------------------------------------
Backbone blocks (exactly as you asked):
  block1: in -> 2048 -> ReLU -> Dropout
  block2: 2048 -> 512 -> ReLU -> Dropout
  block3: 512  -> 256 -> ReLU -> Dropout
  block4: 256  -> 128 -> ReLU -> Dropout
Emotion head uses block4 output (128).
Gender head taps the output of block k (1..4) via GRL.

Data layout (same as your speech code):
  root=/home/lai/SER/models/RAE
    features/fold_1/{risk_train.npy, risk_val.npy, test.npy, scaler.joblib}
    splits/fold_1/{risk_train.csv,  risk_val.csv,  test.csv}
    ...
    features/fold_5/...
    splits/fold_5/...

Exports per fold:
  out_dir/fold_1/embeddings/{risk_train_L{1,2,3,4}.npy, ...}
  out_dir/fold_1/embeddings/{risk_train_meta.csv, ...}

Run (no GRL):
  python speech_grl_mlp.py --root /home/lai/SER/models/RAE --no-grl

Run (GRL at layer 3, logistic schedule):
  python speech_grl_mlp.py --root /home/lai/SER/models/RAE --grl-layer 3 \
      --lambda-schedule logistic --lambda-max 1.0 --lambda-gamma 10.0 --adv-weight 1.0
"""

import argparse, math, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.utils.data import TensorDataset

import csv, os, matplotlib
matplotlib.use("Agg") 

# -------------------- Dataset --------------------
KEEP_EMOS = ["neu", "hap", "sad", "ang"]
EMO_LABELS = {lab: idx for idx, lab in enumerate(KEEP_EMOS)}

class FeatDataset(Dataset):
    def __init__(self, np_path: Path, csv_path: Path):
        raw = np.load(np_path)                       # (N, 6373)
        scaler_path = np_path.with_suffix('').parent / 'scaler.joblib'
        scaler = load(scaler_path)
        feats = scaler.transform(raw).astype(np.float32)

        df = pd.read_csv(csv_path)
        keep_mask = df["emotion"].isin(EMO_LABELS)
        df_kept = df[keep_mask]

        self.x = feats[keep_mask.values]
        self.gender = df_kept["gender"].map({"male":0, "female":1}).values.astype(np.int64)
        self.emotion= df_kept["emotion"].map(EMO_LABELS).values.astype(np.int64)
        self.n_emo = len(EMO_LABELS)

        assert set(np.unique(self.gender)) <= {0,1}, "gender not 0/1"

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]),
                torch.tensor(self.gender[idx]),
                torch.tensor(self.emotion[idx]))


def make_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    w = counts.sum() / (counts + 1e-6) / n_classes
    return torch.tensor(w, dtype=torch.float32)


# -------------------- GRL --------------------
class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = float(lambd); return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out): return (-ctx.lambd) * grad_out, None

def grad_reverse(x, lambd=1.0): return GradReverseFn.apply(x, lambd)

def make_lambda_scheduler(total_steps, lam_max=1.0, gamma=10.0,
                          schedule="logistic", warmup_steps=0):
    step = 0
    def _sched():
        nonlocal step; step += 1
        if step <= warmup_steps: return 0.0
        p = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        p = min(max(p, 0.0), 1.0)
        if schedule == "linear":
            lam = lam_max * p
        elif schedule == "cosine":
            lam = lam_max * 0.5 * (1.0 - math.cos(math.pi * p))
        elif schedule == "const":
            lam = lam_max
        else:  # logistic (DANN)
            lam = lam_max * (2.0/(1.0 + math.exp(-gamma * p)) - 1.0)
        return float(lam)
    return _sched


# -------------------- 4-layer MLP --------------------
class MLP(nn.Module):
    """
    blocks: 4 × [Linear -> ReLU -> Dropout]
    hidden_dims = [2048, 512, 256, 128]
    grl_layer ∈ {1,2,3,4} taps the output of that block
    emotion_head uses block4 output (128)
    """
    def __init__(self, in_dim=6373, hidden_dims=[2048, 512, 256, 128],
                 drop=0.3, n_emo=4, grl_layer=4):
        super().__init__()
        assert len(hidden_dims) == 4
        self.hidden_dims = hidden_dims
        self.grl_layer   = int(grl_layer)

        blocks = []
        prev = in_dim
        for h in hidden_dims:
            blocks.append(nn.Sequential(
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(drop)
            ))
            prev = h
        self.blocks = nn.ModuleList(blocks)

        tap_dim = hidden_dims[self.grl_layer - 1]
        self.gender_head = nn.Sequential(
            nn.Linear(tap_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, 2)
        )
        self.emotion_head = nn.Linear(hidden_dims[-1], n_emo)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

        print(f"[MLP] hidden_dims={hidden_dims} | GRL tap=layer {self.grl_layer} "
              f"(dim={tap_dim}) | emotion_head_dim={hidden_dims[-1]}")

    def forward(self, x, grl_lambda=1.0):
        h_list = []
        h = x
        for blk in self.blocks:
            h = blk(h)
            h_list.append(h)
        tap = h_list[self.grl_layer - 1]
        g_in = grad_reverse(tap, grl_lambda) if grl_lambda != 0.0 else tap
        g_logits = self.gender_head(g_in)
        e_logits = self.emotion_head(h_list[-1])
        return g_logits, e_logits

    @torch.no_grad()
    def extract_layers(self, x):
        """Return dict of block outputs (post-activation+dropout)."""
        self.eval()
        outs = {}
        h = x
        for i, (blk, d) in enumerate(zip(self.blocks, self.hidden_dims), start=1):
            h = blk(h)
            outs[f"L{i}"] = h
        outs["Final"] = outs["L4"]
        return outs


# -------------------- train / eval --------------------
def run_epoch(model, loader, crit_g, crit_e, get_lambda=None, optim=None, device="cpu", adv_weight=1.0):
    train = optim is not None
    model.train() if train else model.eval()

    g_true, g_pred = [], []
    e_true, e_pred = [], []
    total_loss = total_loss_g = total_loss_e = 0.0

    with torch.set_grad_enabled(train):
        for x,g,e in loader:
            x,g,e = x.float().to(device), g.to(device), e.to(device)
            lam = get_lambda() if (train and get_lambda is not None) else 0.0
            out_g,out_e = model(x, lam)

            loss_g = crit_g(out_g, g)
            loss_e = crit_e(out_e, e)
            loss = loss_e + adv_weight * loss_g

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            bs = x.size(0)
            total_loss   += loss.item()*bs
            total_loss_g += loss_g.item()*bs
            total_loss_e += loss_e.item()*bs

            g_pred.extend(out_g.argmax(1).cpu()); g_true.extend(g.cpu())
            e_pred.extend(out_e.argmax(1).cpu()); e_true.extend(e.cpu())

    g_acc  = accuracy_score(g_true,g_pred)
    uar    = recall_score(g_true, g_pred, average="macro")
    e_f1   = f1_score(e_true,e_pred,average="macro")
    return total_loss/len(loader.dataset), g_acc, uar, e_f1, total_loss_g/len(loader.dataset), total_loss_e/len(loader.dataset)


@torch.no_grad()
def dump_embeddings(model: MLP, loader: DataLoader, out_dir: Path, tag: str):
    out_emb = out_dir / "embeddings"; out_emb.mkdir(parents=True, exist_ok=True)
    L1_list, L2_list, L3_list, L4_list = [], [], [], []
    for xb, _, _ in loader:
        feats = model.extract_layers(xb.float().to(next(model.parameters()).device))
        L1_list.append(feats["L1"].cpu().numpy())
        L2_list.append(feats["L2"].cpu().numpy())
        L3_list.append(feats["L3"].cpu().numpy())
        L4_list.append(feats["L4"].cpu().numpy())
    L1 = np.concatenate(L1_list, 0); np.save(out_emb / f"{tag}_L1.npy", L1)
    L2 = np.concatenate(L2_list, 0); np.save(out_emb / f"{tag}_L2.npy", L2)
    L3 = np.concatenate(L3_list, 0); np.save(out_emb / f"{tag}_L3.npy", L3)
    L4 = np.concatenate(L4_list, 0); np.save(out_emb / f"{tag}_L4.npy", L4)
    print(f"[dump] {tag}: L1 {L1.shape}, L2 {L2.shape}, L3 {L3.shape}, L4 {L4.shape}")

def _append_run_row(csv_path, row: dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists())
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def _maybe_dump_attacker_embeddings(model, split_dir: Path, out_dir_f: Path,
                                    attacker_names=("attacker_train","attacker_val")):
    """如果 splits 下存在 attacker_* 就导出其嵌入，方便后续攻击者实验。"""
    from sklearn.preprocessing import StandardScaler
    for tag in attacker_names:
        npy = (split_dir.parent.parent/"features"/split_dir.name/f"{'risk_train' if tag=='attacker_train' else 'risk_val'}.npy")
        csvp= split_dir/f"{tag}.csv"
        if not csvp.exists():
            continue
        # 加载特征（用同 fold 的 scaler）
        scaler = load((split_dir.parent.parent/"features"/split_dir.name/"scaler.joblib"))
        X = scaler.transform(np.load(npy)).astype(np.float32)
        df = pd.read_csv(csvp)
        # 只保留与 FeatDataset 一致的筛选（如果你在构造 attacker_* 时也做过emotion过滤，这里就无需再筛）
        if "emotion" in df.columns and hasattr(model, "extract_layers"):
            xb = torch.from_numpy(X)
            ld = DataLoader(TensorDataset(xb, torch.zeros(len(xb)), torch.zeros(len(xb))), batch_size=256, shuffle=False)
            dump_embeddings(model, ld, out_dir_f, tag)
            # 保存 meta
            keep_cols = [c for c in ["image","emotion","gender","id","path"] if c in df.columns]
            df[keep_cols].to_csv(out_dir_f/"embeddings"/f"{tag}_meta.csv", index=False)
            print(f"[dump] attacker split '{tag}' embeddings exported.")

# ====== 进 main() 的每个 fold 末尾（test 评测后）追加： ======
        # 记录一行 run 汇总（便于之后跨 λ / layer 画 trade-off 曲线）
        # row = {
        #     "layer": args.grl_layer,
        #     "lambda": args.lambda_max,                # 实际 sweep 的 λ_max
        #     "schedule": args.lambda_schedule,
        #     "adv_weight": args.adv_weight,
        #     "val_emo_f1": float(best_f1),
        #     "test_emo_f1": float(te_ef1),
        #     "test_emo_acc": float(te_eacc),
        #     "test_gender_acc": float(te_gacc),
        #     "test_gender_uar": float(te_guar),
        #     "epochs": args.epochs,
        #     "bs": args.bs,
        #     "lr": args.lr,
        # }
        # _sum_csv = args.out_dir / "summary.csv"
        # _pd.DataFrame([row]).to_csv(_sum_csv, index=False)
        # print(f"[summary] wrote {_sum_csv}")

        # # （可选）如果存在 attacker_* split，就一起导出嵌入
        # try:
        #     from torch.utils.data import TensorDataset
        #     _maybe_dump_attacker_embeddings(model, split_dir, out_dir_f)
        # except Exception as e:
        #     print(f"[warn] attacker embeddings export skipped: {e}")

def save_meta(split_dir: Path, out_dir_f: Path, name: str):
    """把原 split 的元数据存一份，便于对齐。按存在的列自动挑选。"""
    df_raw = pd.read_csv(split_dir / f"{name}.csv")
    keep = [c for c in ["image","path","id","emotion","gender"] if c in df_raw.columns]
    df_raw[keep].to_csv(out_dir_f / "embeddings" / f"{name}_meta.csv", index=False)

def dump_attacker_if_exists(feat_dir: Path, split_dir: Path, model: nn.Module,
                            out_dir_f: Path, bs: int = 64):
    """
    可选：若存在 attacker_train/attacker_val 的 CSV（且能复用 risk_train/risk_val 的特征），
    就把它们也导出。找不到文件就跳过并打印提示。
    """
    for tag, npy_name in [("attacker_train","risk_train.npy"),
                          ("attacker_val", "risk_val.npy")]:
        csvp = split_dir / f"{tag}.csv"
        npy  = feat_dir / npy_name
        if not (csvp.exists() and npy.exists()):
            print(f"[attacker] skip {tag}: {csvp.exists()=}, {npy.exists()=}")
            continue
        ds = FeatDataset(npy, csvp)
        ld = DataLoader(ds, batch_size=bs, shuffle=False)
        dump_embeddings(model, ld, out_dir_f, tag)
        save_meta(split_dir, out_dir_f, tag)
        print(f"[attacker] dumped {tag} embeddings.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="目录同时包含 features/ 与 splits/ （5 folds）")
    ap.add_argument("--out-dir", type=Path, default=Path("./exp_speech_grl"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)

    # GRL 开关与参数
    ap.add_argument("--no-grl", action="store_true", help="禁用GRL（等价于 adv-weight=0 & lambda=0）")
    ap.add_argument("--grl-layer", type=int, default=4, help="GRL 插入层 1..4")
    ap.add_argument("--adv-weight", type=float, default=1.0, help="gender 对抗损失的权重")
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--lambda-gamma", type=float, default=10.0)
    ap.add_argument("--lambda-schedule", choices=["logistic","linear","cosine","const"], default="logistic")
    ap.add_argument("--warmup-epochs", type=int, default=0)

    args = ap.parse_args()
    root = Path(args.root)
    out_root = args.out_dir; out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    folds = sorted([p.name for p in (root/"features").iterdir() if p.is_dir()],
                   key=lambda x: int(x.split('_')[-1]))
    assert len(folds) == 5, f"check if 5 folds, got {len(folds)}"

    val_fold_scores, test_fold_scores = [], []

    for f in folds:
        print(f"\n===== {f} =====")
        feat_dir  = root/"features"/f
        split_dir = root/"splits"/f
        out_dir_f = out_root/f; out_dir_f.mkdir(parents=True, exist_ok=True)

        tr_ds = FeatDataset(feat_dir/"risk_train.npy", split_dir/"risk_train.csv")
        va_ds = FeatDataset(feat_dir/"risk_val.npy",   split_dir/"risk_val.csv")
        te_ds = FeatDataset(feat_dir/"test.npy",       split_dir/"test.csv")

        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False)
        te_ld = DataLoader(te_ds, batch_size=args.bs, shuffle=False)

        emo_w = make_weights(tr_ds.emotion, n_classes=tr_ds.n_emo).to(device)
        gen_w = make_weights(tr_ds.gender,  n_classes=2).to(device)
        crit_e = nn.CrossEntropyLoss(weight=emo_w)
        crit_g = nn.CrossEntropyLoss(weight=gen_w)

        model = MLP(in_dim=tr_ds.x.shape[1], n_emo=tr_ds.n_emo,
                    grl_layer=args.grl_layer).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

        total_steps  = args.epochs * len(tr_ld)
        warmup_steps = args.warmup_epochs * len(tr_ld)
        get_lambda = make_lambda_scheduler(total_steps, args.lambda_max,
                                           args.lambda_gamma, args.lambda_schedule, warmup_steps)

        adv_w = 0.0 if args.no_grl else args.adv_weight
        if args.no_grl:
            print("[GRL] disabled: adv_weight=0, lambda schedule ignored.")
        else:
            print(f"[GRL] layer={args.grl_layer} schedule={args.lambda_schedule} "
                  f"lam_max={args.lambda_max} gamma={args.lambda_gamma} warmup_epochs={args.warmup_epochs}")

        best_val = (-1.0, None)  # (val_f1, state)
        for ep in range(1, args.epochs + 1):
            tr = run_epoch(model, tr_ld, crit_g, crit_e,
                           get_lambda=(None if args.no_grl else get_lambda),
                           optim=optim, device=device, adv_weight=adv_w)
            va = run_epoch(model, va_ld, crit_g, crit_e,
                           get_lambda=None, optim=None, device=device, adv_weight=adv_w)
            (tr_loss, tr_g, tr_uar, tr_f1, tr_lg, tr_le) = tr
            (va_loss, va_g, va_uar, va_f1, va_lg, va_le) = va
            print(f"E{ep:02d} tr-loss {tr_loss:.3f} (g {tr_lg:.3f}/e {tr_le:.3f}) | "
                  f"tr-Gacc {tr_g:.3f} UAR {tr_uar:.3f} Ef1 {tr_f1:.3f}")
            print(f"      va-loss {va_loss:.3f} (g {va_lg:.3f}/e {va_le:.3f}) | "
                  f"va-Gacc {va_g:.3f} UAR {va_uar:.3f} Ef1 {va_f1:.3f}")

            if va_f1 > best_val[0]:
                best_val = (va_f1, {k: v.detach().cpu() for k, v in model.state_dict().items()})

        assert best_val[1] is not None, "No best state saved."
        torch.save(best_val[1], out_dir_f / f"mlp_best_L{args.grl_layer}_{'noGRL' if args.no_grl else args.lambda_schedule}.pt")

        model.load_state_dict(best_val[1], strict=True)
        te = run_epoch(model, te_ld, crit_g, crit_e, get_lambda=None, optim=None, device=device, adv_weight=adv_w)
        _, test_g, test_uar, test_f1, _, _ = te

        print(f"{f}: best-VAL Ef1={best_val[0]:.4f} | TEST Gacc={test_g:.4f} UAR={test_uar:.4f} Ef1={test_f1:.4f}")
        val_fold_scores.append(best_val[0]); test_fold_scores.append((test_g, test_uar, test_f1))

        # ---- 导出四层 embedding（供 attacker）----
        # ---- 导出四层 embedding（供 attacker 使用）----
        dump_embeddings(model, tr_ld, out_dir_f, "risk_train")
        dump_embeddings(model, va_ld, out_dir_f, "risk_val")
        dump_embeddings(model, te_ld, out_dir_f, "test")

        # 对应 meta
        for name in ["risk_train", "risk_val", "test"]: 
            save_meta(split_dir, out_dir_f, name)

        # （可选）若存在 attacker_* 划分，也一起导出
        dump_attacker_if_exists(feat_dir, split_dir, model, out_dir_f, bs=args.bs)

        # 保存对应的 meta（id/emotion/gender）方便对齐
        for name in ["risk_train", "risk_val", "test"]:
            df = pd.read_csv(split_dir / f"{name}.csv")[["id","emotion","gender"]] if "id" in pd.read_csv(split_dir / f"{name}.csv").columns \
                 else pd.read_csv(split_dir / f"{name}.csv")[["emotion","gender"]].assign(id=np.arange(len(pd.read_csv(split_dir / f"{name}.csv"))))
            df.to_csv(out_dir_f / "embeddings" / f"{name}_meta.csv", index=False)

    print("\n========== 5-fold summary ==========")
    print(f"VAL  Emotion-F1 (mean): {np.mean(val_fold_scores):.4f}")
    print(f"TEST Gender-Acc (mean): {np.mean([x[0] for x in test_fold_scores]):.4f}")
    print(f"TEST Gender-UAR (mean): {np.mean([x[1] for x in test_fold_scores]):.4f}")
    print(f"TEST Emotion-F1 (mean): {np.mean([x[2] for x in test_fold_scores]):.4f}")
    print("====================================")

    # ==== 写一行 run 级别 summary.csv（供 sweep 合并画图）====
    val_emo_f1_mean   = float(np.mean(val_fold_scores))
    test_gender_acc_m = float(np.mean([x[0] for x in test_fold_scores]))
    test_gender_uar_m = float(np.mean([x[1] for x in test_fold_scores]))
    test_emo_f1_mean  = float(np.mean([x[2] for x in test_fold_scores]))

    row = {
        "layer": args.grl_layer,
        "lambda": args.lambda_max,
        "schedule": ("none" if args.no_grl else args.lambda_schedule),
        "adv_weight": (0.0 if args.no_grl else args.adv_weight),
        "val_emo_f1": val_emo_f1_mean,
        "test_emo_f1": test_emo_f1_mean,
        "test_emo_acc": None,                 # 如果你也想存 test 情绪 acc，可在 run_epoch 里返回 e_acc
        "test_gender_acc": test_gender_acc_m,
        "test_gender_uar": test_gender_uar_m,
        "epochs": args.epochs,
        "bs": args.bs,
        "lr": args.lr,
    }
    summary_csv = out_root / "summary.csv"
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    print(f"[summary] wrote {summary_csv}")


if __name__ == "__main__":
    main()
