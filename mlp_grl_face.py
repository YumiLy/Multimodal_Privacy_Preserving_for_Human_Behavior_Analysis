#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRL + 4-layer MLP for Emotion (CNN features)
--------------------------------------------
Backbone: 2048 -> 512 -> 256 -> 128 (ReLU + BN + Dropout=0.3)
Heads   : emotion_head(128 -> n_cls), gender_head(tap_dim -> 2) via GRL

CLI:
  python grl_mlp_emotion.py \
    --split-dir /home/lai/FER/data/splits_features \
    --out-dir   /home/lai/FER/experiments/grl_mlp \
    --epochs 60 --bs 256 --lr 3e-4 --seed 42 --device auto \
    --grl-layer 3 --lambda-schedule logistic --lambda-max 1.0 --lambda-gamma 10.0 --warmup-epochs 0 \
    --adv-weight 1.0

Notes:
- Only computes gender loss/metrics on rows with known gender (male/female). Unknowns are masked.
- Standardizes features with StandardScaler (fit on train, apply to val/test).
"""

import argparse, math, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from joblib import dump
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import csv, os, time
from datetime import datetime

def append_summary_row(csv_path, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists())
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

# ---------------- Utils ----------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_split(split_dir: Path, tag: str, emo_map=None):
    X = np.load(split_dir / f"{tag}_feats.npy")
    df = pd.read_csv(split_dir / f"{tag}_labels.csv")
    y_e_raw = df["emotion"].astype(int).values if "emotion" in df.columns else df["label"].astype(int).values

    if emo_map is None:
        uniq = sorted(np.unique(y_e_raw))
        emo_map = {lab:i for i,lab in enumerate(uniq)}
        print(f"[{tag}] emotion map:", emo_map)
    y_e = np.array([emo_map[int(l)] for l in y_e_raw], dtype=np.int64)

    g_raw = df["gender"].astype(str).str.lower().values if "gender" in df.columns else df["perceived_gender"].astype(str).str.lower().values
    map_g = {"male":0, "female":1}
    y_g = np.array([map_g[g] if g in map_g else -1 for g in g_raw], dtype=np.int64)
    return X, y_e, y_g, emo_map

# --------------- Dataset ---------------
class FeatDS(Dataset):
    def __init__(self, X: np.ndarray, y_e: np.ndarray, y_g: np.ndarray):
        assert len(X)==len(y_e)==len(y_g)
        self.x = torch.from_numpy(X.astype(np.float32))
        self.ye= torch.from_numpy(y_e.astype(np.int64))
        self.yg= torch.from_numpy(y_g.astype(np.int64))
    def __len__(self): return len(self.ye)
    def __getitem__(self, i): return self.x[i], self.yg[i], self.ye[i]

# --------- Gradient Reversal Layer ----------
class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out): return (-ctx.lambd) * grad_out, None
def grad_reverse(x, lambd=1.0): return GradReverseFn.apply(x, lambd)

def make_lambda_scheduler(total_steps, lam_max=1.0, gamma=10.0, schedule="logistic", warmup_steps=0):
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

# --------------- Model -----------------
class MLP(nn.Module):
    """
    blocks: 4 × [Linear -> ReLU -> Dropout]
    hidden_dims = [2048, 512, 256, 128]
    - grl_layer ∈ {1,2,3,4}: 在该 block 的 **输出** 处接 GRL+gender 头
    - emotion_head 始终用第4个 block 的输出（128）
    """
    def __init__(self, in_dim=6373, hidden_dims=[2048, 512, 256, 128],
                 drop=0.3, n_emo=4, grl_layer=4, use_bn=False):
        super().__init__()
        assert len(hidden_dims) == 4, "需要4层隐藏层 [2048,512,256,128]"
        self.hidden_dims = hidden_dims
        self.grl_layer   = int(grl_layer)

        # 构建4个 block
        blocks = []
        prev = in_dim
        for h in hidden_dims:
            layers = [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if use_bn: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(drop))
            blocks.append(nn.Sequential(*layers))
            prev = h
        self.blocks = nn.ModuleList(blocks)

        # 性别头接在 tap_dim 上
        assert 1 <= self.grl_layer <= 4
        tap_dim = hidden_dims[self.grl_layer - 1]
        self.gender_head = nn.Sequential(
            nn.Linear(tap_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, 2)
        )

        # 情感头接在最后一层（128）
        self.emotion_head = nn.Linear(hidden_dims[-1], n_emo)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

        # 运行时提示，避免 tap 混淆
        print(f"[MLP] hidden_dims={hidden_dims} | GRL tap=layer {self.grl_layer} "
              f"(dim={tap_dim}) | emotion_head_dim={hidden_dims[-1]}")

    def forward(self, x, grl_lambda=1.0):
        # 依次通过4个 block，并在指定层取 tap
        h_list = []
        h = x
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h)          # 经过 Linear+ReLU(+BN)+Dropout
            h_list.append(h)    # h_list[i-1] 即第 i 层输出

        tap = h_list[self.grl_layer - 1]              # 第 k 层输出
        g_in = grad_reverse(tap, grl_lambda) if grl_lambda != 0.0 else tap
        g_logits = self.gender_head(g_in)

        e_logits = self.emotion_head(h_list[-1])      # 最后一层输出 -> emotion
        return g_logits, e_logits

    @torch.no_grad()
    def extract_layers(self, x):
        """
        返回4层的输出特征（均为 post-activation+dropout 之后）：
        L1: 2048, L2: 512, L3: 256, L4: 128
        """
        self.eval()
        outs = {}
        h = x
        for i, (blk, d) in enumerate(zip(self.blocks, self.hidden_dims), start=1):
            h = blk(h)
            outs[f"L{i}"] = h  # 形状 [N, d]
        outs["Final"] = outs["L4"]            # 兼容以前的命名
        return outs

# -------- Masked CE for gender (ignore unknown) --------
def masked_ce(logits, targets, mask, weight=None):
    """
    logits: [B, C], targets: [B], mask: [B] bool/int (1=known)
    returns scalar loss averaged over known samples (safe if none known)
    """
    if mask.sum().item() == 0:
        return logits.new_tensor(0.0)
    crit = nn.CrossEntropyLoss(weight=weight, reduction="none")
    loss_vec = crit(logits, targets)
    loss = (loss_vec * mask.float()).sum() / mask.float().sum()
    return loss

# --------------- Train/Eval loops ---------------
def run_epoch(model, loader, crit_e, w_g=None, get_lambda=None, optim=None, device="cpu", adv_weight=1.0):
    train = optim is not None
    model.train() if train else model.eval()

    g_true, g_pred = [], []
    e_true, e_pred = [], []
    loss_sum = loss_e_sum = loss_g_sum = 0.0
    with torch.set_grad_enabled(train):
        for xb, yg, ye in loader:
            xb, yg, ye = xb.to(device), yg.to(device), ye.to(device)
            lam = get_lambda() if (train and get_lambda is not None) else 0.0
            g_logit, e_logit = model(xb, lam)

            # emotion loss
            loss_e = crit_e(e_logit, ye)

            # gender loss (mask unknown = -1)
            known = (yg >= 0)
            # clamp targets to valid (avoid CE complaining when no known)
            yg_clamp = torch.clamp(yg, 0, 1)
            loss_g = masked_ce(g_logit, yg_clamp, known, weight=w_g)

            loss = loss_e + adv_weight * loss_g

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            # stats
            bs = xb.size(0)
            loss_sum  += loss.item()  * bs
            loss_e_sum+= loss_e.item()* bs
            loss_g_sum+= loss_g.item()* bs

            e_true.append(ye.cpu()); e_pred.append(e_logit.argmax(1).cpu())
            if known.any():
                g_true.append(yg[known].cpu()); g_pred.append(g_logit[known].argmax(1).cpu())

    N = len(loader.dataset)
    e_y = torch.cat(e_true).numpy(); e_p = torch.cat(e_pred).numpy()
    e_f1 = f1_score(e_y, e_p, average="macro")
    e_acc= accuracy_score(e_y, e_p)

    if len(g_true) > 0:
        g_y = torch.cat(g_true).numpy(); g_p = torch.cat(g_pred).numpy()
        g_acc = accuracy_score(g_y, g_p)
        g_uar = recall_score(g_y, g_p, average="macro")
        g_cov = len(g_y) / N
    else:
        g_acc = g_uar = 0.0; g_cov = 0.0

    return (loss_sum/N, loss_g_sum/N, loss_e_sum/N,
            g_acc, g_uar, g_cov, e_acc, e_f1)

def dump_embeddings(model, split_dir: Path, tag: str, loader: DataLoader,
                    out_root: Path, emo_map: dict):
    """
    导出某个 split 的各层嵌入到 .npy，并保存对齐的 id/emotion/gender 到 .csv
    - 保存路径: <out_root>/embeddings/<tag>_{L1,L2,L3,Final}.npy
    - 对应标签: <out_root>/embeddings/<tag}_meta.csv  (image, emotion, gender)
    """
    import numpy as np, pandas as pd, os
    model.eval()
    out_dir = out_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 split 的原始 labels.csv 以拿到 image/path/gender
    meta_csv = split_dir / f"{tag}_labels.csv"
    meta_df  = pd.read_csv(meta_csv)  # 期望有: image/path, emotion(原始), gender
    # emotion 已在训练中重映射，这里也生成一列 remapped 以与 .npy 对齐
    # 处理可能的缺列名差异
    if "emotion" not in meta_df.columns and "label" in meta_df.columns:
        meta_df = meta_df.rename(columns={"label":"emotion"})
    meta_df["emotion_remap"] = meta_df["emotion"].map(emo_map).astype(int)

    # 遍历 loader 收集各层
    L1_list, L2_list, L3_list, F_list = [], [], [], []
    with torch.no_grad():
        for xb, yg, ye in loader:  # xb 已在 main 中标准化
            xb = xb.to(next(model.parameters()).device)
            feats = model.extract_layers(xb)
            L1_list.append(feats["L1"].cpu().numpy())
            L2_list.append(feats["L2"].cpu().numpy())
            L3_list.append(feats["L3"].cpu().numpy())
            F_list.append(feats["Final"].cpu().numpy())

    L1 = np.concatenate(L1_list, axis=0)
    L2 = np.concatenate(L2_list, axis=0)
    L3 = np.concatenate(L3_list, axis=0)
    FF = np.concatenate(F_list,  axis=0)

    # 安全起见，检查长度是否与 meta 对齐
    assert len(meta_df) == len(FF), f"length mismatch: meta={len(meta_df)} vs feats={len(FF)}"

    # 保存
    np.save(out_dir / f"{tag}_L1.npy", L1)
    np.save(out_dir / f"{tag}_L2.npy", L2)
    np.save(out_dir / f"{tag}_L3.npy", L3)
    np.save(out_dir / f"{tag}_Final.npy", FF)
    meta_keep = meta_df[["image", "emotion_remap", "gender"]].copy() if "image" in meta_df.columns \
                else meta_df[["path",  "emotion_remap", "gender"]].rename(columns={"path":"image"})
    meta_keep.to_csv(out_dir / f"{tag}_meta.csv", index=False)

    print(f"[dump] {tag}: saved to {out_dir} (L1:{L1.shape}, L2:{L2.shape}, L3:{L3.shape}, Final:{FF.shape})")


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, required=True, help="dir with *_feats.npy and *_labels.csv")
    ap.add_argument("--out-dir",   type=Path, required=True)
    ap.add_argument("--epochs",    type=int, default=60)
    ap.add_argument("--bs",        type=int, default=256)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--drop",      type=float, default=0.3)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--device",    type=str, default="auto", choices=["auto","cuda","cpu"])

    # GRL / schedule
    ap.add_argument("--grl-layer", type=int, default=3, help="insert GRL after this layer (1..4)")
    ap.add_argument("--lambda-schedule", choices=["logistic","linear","cosine","const"], default="logistic")
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--lambda-gamma", type=float, default=10.0)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--adv-weight", type=float, default=1.0, help="weight of gender loss term")

    ap.add_argument("--patience",  type=int, default=8, help="early stop on val emotion F1")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device
    print(f"Device: {device}")

    split_dir = args.split_dir
    # Load splits
    Xtr, ytr_e, ytr_g, emo_map = load_split(split_dir, "main_train", emo_map=None)
    Xva, yva_e, yva_g, _ = load_split(split_dir, "main_val", emo_map=emo_map)
    Xte, yte_e, yte_g, _ = load_split(split_dir, "test", emo_map=emo_map)

    print(f"Shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")
    n_classes = int(max(ytr_e.max(), yva_e.max(), yte_e.max()) + 1)
    in_dim    = Xtr.shape[1]

    # Standardize features (fit on train only)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)
    dump(scaler, args.out_dir / "scaler.joblib")
    print("Saved scaler:", args.out_dir/"scaler.joblib")

    # Datasets/Loaders
    ds_tr = FeatDS(Xtr_s, ytr_e, ytr_g); ds_va = FeatDS(Xva_s, yva_e, yva_g); ds_te = FeatDS(Xte_s, yte_e, yte_g)
    ld_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,  drop_last=False)
    ld_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, drop_last=False)
    ld_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, drop_last=False)

    # Class weights (emotion from train, gender from known train)
    emo_counts = np.bincount(ytr_e, minlength=n_classes).astype(float)
    emo_w = torch.tensor(emo_counts.sum() / (emo_counts + 1e-6) / n_classes, dtype=torch.float32, device=device)
    g_known = ytr_g[ytr_g >= 0]
    if len(g_known) > 0:
        g_counts = np.bincount(g_known, minlength=2).astype(float)
        g_w = torch.tensor(g_counts.sum() / (g_counts + 1e-6) / 2, dtype=torch.float32, device=device)
    else:
        g_w = None

    crit_e = nn.CrossEntropyLoss(weight=emo_w)

    # Build model
    model = MLP(in_dim=in_dim, drop=args.drop, n_emo=n_classes, grl_layer=args.grl_layer).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps  = args.epochs * max(1, len(ld_tr))
    warm_steps   = args.warmup_epochs * max(1, len(ld_tr))
    get_lambda = make_lambda_scheduler(total_steps, args.lambda_max, args.lambda_gamma, args.lambda_schedule, warm_steps)
    print(f"GRL: layer={args.grl_layer}  sched={args.lambda_schedule}  lam_max={args.lambda_max}  gamma={args.lambda_gamma}  warmup_epochs={args.warmup_epochs}")

    # Train
    best_f1 = -1.0
    best_state = None
    bad = 0
    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, ld_tr, crit_e, w_g=g_w, get_lambda=get_lambda, optim=optim, device=device, adv_weight=args.adv_weight)
        va = run_epoch(model, ld_va, crit_e, w_g=g_w, get_lambda=None,   optim=None,  device=device, adv_weight=args.adv_weight)
        (tr_loss, tr_gl, tr_el, tr_gacc, tr_guar, tr_gcov, tr_eacc, tr_ef1) = tr
        (va_loss, va_gl, va_el, va_gacc, va_guar, va_gcov, va_eacc, va_ef1) = va
        print(f"Ep{ep:02d} | tr: L {tr_loss:.4f} (g {tr_gl:.4f} / e {tr_el:.4f}) | g-acc {tr_gacc:.3f} uar {tr_guar:.3f} cov {tr_gcov:.2f} | e-acc {tr_eacc:.3f} f1 {tr_ef1:.3f}")
        print(f"        va: L {va_loss:.4f} (g {va_gl:.4f} / e {va_el:.4f}) | g-acc {va_gacc:.3f} uar {va_guar:.3f} cov {va_gcov:.2f} | e-acc {va_eacc:.3f} f1 {va_ef1:.3f}")

        if va_ef1 > best_f1:
            best_f1 = va_ef1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep}. Best val F1 = {best_f1:.4f}")
                break

    # Save & test
    assert best_state is not None, "No best_state saved; check training."
    ckpt = args.out_dir / f"grl_mlp_best_L{args.grl_layer}_{args.lambda_schedule}_lam{args.lambda_max}_g{args.lambda_gamma}.pt"
    torch.save(best_state, ckpt)
    print("Saved best model to:", ckpt)

    model.load_state_dict(best_state)
     # ==== Dump embeddings for attacker ====
    # 注意：用与训练相同的 scaler 标准化后的特征 (loader 里的 xb 已经是标准化结果)
    dump_embeddings(model, split_dir, "main_train", ld_tr, args.out_dir, emo_map)
    dump_embeddings(model, split_dir, "main_val",   ld_va, args.out_dir, emo_map)
    dump_embeddings(model, split_dir, "test",       ld_te, args.out_dir, emo_map)


    te = run_epoch(model, ld_te, crit_e, w_g=g_w, get_lambda=None, optim=None, device=device, adv_weight=args.adv_weight)
    (te_loss, te_gl, te_el, te_gacc, te_guar, te_gcov, te_eacc, te_ef1) = te
    print(f"[TEST] L {te_loss:.4f} (g {te_gl:.4f} / e {te_el:.4f}) | g-acc {te_gacc:.3f} uar {te_guar:.3f} cov {te_gcov:.2f} | e-acc {te_eacc:.3f} f1 {te_ef1:.3f}")

        # === 写入 summary.csv（只记录 TEST 指标 + 关键超参，用于画图）===
    summary_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layer": int(args.grl_layer),
        "lambda_max": float(args.lambda_max),
        "lambda_schedule": str(args.lambda_schedule),
        "lambda_gamma": float(args.lambda_gamma),
        "adv_weight": float(args.adv_weight),
        "epochs": int(args.epochs),
        "batch_size": int(args.bs),
        "lr": float(args.lr),
        "seed": int(args.seed),

        # TEST metrics you asked for:
        "test_gender_acc": float(te_gacc),
        "test_gender_uar": float(te_guar),
        "test_emotion_acc": float(te_eacc),
        "test_emotion_f1": float(te_ef1),
    }
    append_summary_row(args.out_dir / "summary__all_layers.csv", summary_row)
    print(f"[summary] appended to {args.out_dir / 'summary__all_layers.csv'}")


    # 可选打印每类报告（情绪）
    # 需要 logits/预测明细再写一个 eval_detail；这里简化为指标

if __name__ == "__main__":
    main()
