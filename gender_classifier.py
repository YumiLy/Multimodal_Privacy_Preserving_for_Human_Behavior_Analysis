#!/usr/bin/env python3
"""
Multi‑task gender + emotion classifier
———————————————
* 输入: 6373 维 ComParE‑2013 功能级特征
* 主干: VGG16‑1D (Conv1d) + 两个线性任务头
* 评估: 5‑fold StratifiedKFold, 报平均指标
"""
print("Running multi‑task training...")
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import load 
from sklearn.metrics import recall_score

KEEP_EMOS = ["neu", "hap", "sad", "ang"]       # used {} set before!!! use list instead
EMO_LABELS = {lab: idx for idx, lab in enumerate(KEEP_EMOS)}
# --------------- 数据集 --------------- #
class FeatDataset(Dataset):
    def __init__(self, np_path, csv_path):
        # 全加载 .npy
        raw = np.load(np_path)          # shape = (N, 6373)

        # load the scaler that lives in the *same* fold directory
        scaler_path = Path(np_path).with_suffix('').parent / 'scaler.joblib'
        self.scaler = load(scaler_path)
        feats = self.scaler.transform(raw).astype(np.float32)   # z‑score 

        # 读 CSV 并做过滤
        df = pd.read_csv(csv_path)
        keep_mask = df["emotion"].isin(EMO_LABELS)

        # 同步裁剪特征 & 标签
        self.x       = feats[keep_mask.values]
        df_kept      = df[keep_mask]

        self.gender  = df_kept["gender"].map({"male":0, "female":1}).values

        print(f"[{csv_path.stem}] kept {len(self.x)}/{len(feats)} samples")

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]),            # float32, (6373,)
                torch.tensor(self.gender[idx]))          # int64

# --------------- 权重计算 --------------- #
def make_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    # 避免除 0，且让均值=1
    weights = counts.sum() / (counts + 1e-6) / n_classes
    return torch.tensor(weights, dtype=torch.float32)

# --------------- VGG16‑1D --------------- #
def vgg1d_make_layers(cfg, in_channels=1):
    layers=[]
    for v in cfg:
        if v=='M':
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv1d(in_channels, v, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels=v
    return nn.Sequential(*layers)

class VGG16_1D(nn.Module):
    """
    1‑D 版本的 VGG‑16:
    conv 配置与官方一致: [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
    """
    def __init__(self, in_len=6373, base_channels=64, drop=0.2):
        super().__init__()
        # 若想整体缩小通道数，可把 base_channels 改成 32 或 16
        mul          = base_channels // 64
        cfg_original = [64,64,'M',128,128,'M',256,256,256,'M',
                        512,512,512,'M',512,512,512,'M']
        cfg = [v*mul if isinstance(v,int) else v for v in cfg_original]

        self.features = vgg1d_make_layers(cfg, in_channels=1)          # (N,1,L) -> (N,C,L')
        # 自适应池化到固定长度，方便后续全连接
        self.gap      = nn.AdaptiveAvgPool1d(1)                        # (N,C,1)
        self.flatten  = nn.Flatten()
        hidden        = 512*mul
        self.dropout  = nn.Dropout(drop)
        self.input_bn = nn.BatchNorm1d(1, affine=False)                       # (N,C)

        self.gender_head = nn.Linear(hidden, 2)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):                # x:(N,6373)
        x = x.unsqueeze(1)               # -> (N,1,6373)
        # print("mean", x.mean().item(), "std", x.std().item()); exit()
        x = self.input_bn(x)
        x = self.features(x)
        x = self.gap(x)                  # -> (N,C,1)
        x = self.flatten(x)              # -> (N,C)
        x = self.dropout(x)
        return self.gender_head(x)


class MLP(nn.Module):
    def __init__(self, in_dim=6373, hidden1=2048, hidden2=512, drop=0.3, n_gender=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.BatchNorm1d(hidden1),
            nn.Dropout(drop),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.BatchNorm1d(hidden2),
            nn.Dropout(drop),
        )
        self.gender_head = nn.Linear(hidden2, n_gender)

        for m in self.modules():                 # 初始化
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):                        # x:(N, 6373)
        h = self.backbone(x)
        return self.gender_head(h)


# --------------- 训练 / 验证 --------------- #
def run_epoch(model, loader, crit_g, optim=None, device="cpu"):
    train = optim is not None
    model.train() if train else model.eval()

    g_true, g_pred = [], []
    total_loss = 0
    with torch.set_grad_enabled(train):
        for x,g in loader:
            x,g = x.float().to(device), g.to(device)
            out_g = model(x)
            loss = crit_g(out_g,g)

            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()*x.size(0)
            g_pred.extend(out_g.argmax(1).cpu()); g_true.extend(g.cpu())

    g_acc   = accuracy_score(g_true,g_pred)
    uar = recall_score(g_true, g_pred, average="macro")
    return total_loss/len(loader.dataset), g_acc, uar

# --------------- 主程序 --------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="目录同时包含 features/ 与 splits/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=64, help="VGG 通道基数 64/32/16…")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading data from:", args.root)
    root = Path(args.root)
    feat_root, split_root = root/"features", root/"splits"
    folds = sorted([p.name for p in feat_root.iterdir() if p.is_dir()])
    assert len(folds) == 5, f"check if 5 folds, {len(folds)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # crit   = nn.CrossEntropyLoss()
    

    # ------------------------------------------------------------------ #
    # 1) mapping emotion labels
    # ------------------------------------------------------------------ #
    print("Building emotion mapping…")
    all_emo = set()
    for f in folds:
        for csv_name in ["risk_train.csv", "risk_val.csv", "test.csv"]:
            df = pd.read_csv(split_root/f/csv_name)
            all_emo.update(df["emotion"].unique())
    print("All emotion labels:", all_emo)
    assert all_emo.issubset(KEEP_EMOS), "Some emotion labels are not in EMO_LABELS"
    EMO_MAP = {lab: idx for idx, lab in enumerate(sorted(KEEP_EMOS))}
    globals()["EMO_MAP"] = EMO_MAP         # 覆盖全局，供 Dataset 默认使用
    print("Emotion mapping:", EMO_MAP)

    # ------------------------------------------------------------------ #
    # 2) Do 5-fold training
    # ------------------------------------------------------------------ #
    val_fold_scores  = []   # [(best_val_g, best_val_f1), ...]
    test_fold_scores = []   # [(test_g, test_f1), ...]

    for f in folds:
        print(f"\n===== {f} =====")
        feat_dir  = feat_root/f
        split_dir = split_root/f

        tr_ds = FeatDataset(feat_dir/"risk_train.npy", split_dir/"risk_train.csv")
        va_ds = FeatDataset(feat_dir/"risk_val.npy", split_dir/"risk_val.csv")
        te_ds = FeatDataset(feat_dir/"test.npy", split_dir/"test.csv")

        # emo_w = make_weights(tr_ds.emotion, n_classes=4).to(device)
        gender_w = make_weights(tr_ds.gender,  n_classes=2).to(device)  # 可选
        # print(f"Emotion weights: {emo_w}")
        print(f"Gender weights: {gender_w}")

        # crit_e = nn.CrossEntropyLoss(weight=emo_w)
        crit_g = nn.CrossEntropyLoss(weight=gender_w)

        # n_emo 取训练集（也可 len(EMO_MAP)，两者应一致）
        # n_emo = tr_ds.n_emo

        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=args.bs)
        te_ld = DataLoader(te_ds, batch_size=args.bs)

        # model = VGG16_1D(base_channels=args.base, n_emo_classes=n_emo).to(device)
        model = MLP().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_val_g   = 0.0
        best_val_uar = 0.0
        best_state   = None

        for ep in range(1, args.epochs + 1):
            run_epoch(model, tr_ld, crit_g, optim, device)               # train step
            _, val_g, val_uar = run_epoch(model, va_ld, crit_g, None, device)  # val step

            # 如果验证提升则记录权重
            if val_g > best_val_g:
                best_val_g = val_g
                best_val_uar = val_uar
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"E{ep:02d} val-Gacc {val_g:.3f} val-UAR {val_uar:.3f}", end="\r")
        print()  # 换行

        # 使用验证集最佳模型评测 test
        assert best_state is not None, "best_state 未设置（检查训练循环）"
        model.load_state_dict(best_state, strict=True)
        _, test_g, test_uar = run_epoch(model, te_ld, crit_g, None, device)

        print(f"{f}: best-VAL Gacc={best_val_g:.4f} | "
              f"TEST Gacc={test_g:.4f} | TEST UAR={test_uar:.4f}")

        val_fold_scores.append((best_val_g, best_val_uar))
        test_fold_scores.append((test_g, test_uar))

        # 可选：保存 best 权重
        torch.save(best_state, f"mlp_{f}_best_gender_only.pt")

    # ------------------------------------------------------------------ #
    # 3) 汇总 5 折
    # ------------------------------------------------------------------ #
    val_g_mean = np.mean([s[0] for s in val_fold_scores])
    val_uar_mean = np.mean([s[1] for s in val_fold_scores])
    test_g_mean = np.mean([s[0] for s in test_fold_scores])
    test_uar_mean = np.mean([s[1] for s in test_fold_scores])

    print("\n========== 5-fold results ==========")
    print(f"VAL  Gender-Acc (mean): {val_g_mean:.4f}")
    print(f"VAL  Gender-UAR (mean): {val_uar_mean:.4f}")
    print(f"TEST Gender-Acc (mean): {test_g_mean:.4f}")
    print(f"TEST Gender-UAR (mean): {test_uar_mean:.4f}")
    print("====================================")


if __name__ == "__main__":
    main()