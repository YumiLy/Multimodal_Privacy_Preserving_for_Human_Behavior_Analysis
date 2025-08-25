#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export embeddings for chosen layers from a trained speech MLP model.

Use-cases:
- Export Baseline (no-GRL) model embeddings for attacker pools.
- Export Privatized (L4, lambda=0.6) model embeddings for attacker/main/test.

You can map source split file names (e.g., risk_train/risk_val/test) to
target embedding tags (e.g., trans_train/trans_val/test) via --from-tags/--to-tags.

Example (baseline -> trans_*):
  python export_embeddings_from_ckpt.py \
    --root /home/lai/SER/models/RAE \
    --exp  /home/lai/SER/experiments/speech_grl/no_grl \
    --grl-layer 4 \
    --from-tags risk_train,risk_val,test \
    --to-tags   trans_train,trans_val,test \
    --layers L1,L2,L3

Example (priv λ=0.6 L4 -> trans_*):
  python export_embeddings_from_ckpt.py \
    --root /home/lai/SER/models/RAE \
    --exp  /home/lai/SER/experiments/speech_grl/layer4_lam0.6 \
    --grl-layer 4 \
    --from-tags risk_train,risk_val,test \
    --to-tags   trans_train,trans_val,test \
    --layers L1,L2,L3
"""
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from joblib import load
from torch.utils.data import DataLoader, TensorDataset

KEEP_EMOS = ["neu","hap","sad","ang"]
EMO_LABELS = {lab:i for i,lab in enumerate(KEEP_EMOS)}

class MLP(nn.Module):
    def __init__(self, in_dim=6373, hidden_dims=[2048,512,256,128], drop=0.3, n_emo=4, grl_layer=4):
        super().__init__()
        self.hidden_dims = hidden_dims; self.grl_layer=int(grl_layer)
        blks=[]; prev=in_dim
        for h in hidden_dims:
            blks.append(nn.Sequential(nn.Linear(prev,h), nn.ReLU(inplace=True), nn.Dropout(drop)))
            prev=h
        self.blocks = nn.ModuleList(blks)
        self.gender_head = nn.Sequential(nn.Linear(hidden_dims[self.grl_layer-1],128),
                                         nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(128,2))
        self.emotion_head = nn.Linear(hidden_dims[-1], n_emo)
    @torch.no_grad()
    def extract_layers(self, x):
        outs={}; h=x
        for i,blk in enumerate(self.blocks, start=1):
            h=blk(h); outs[f"L{i}"]=h
        return outs  # L1..L4

def _list_folds(root: Path):
    folds=[p.name for p in (root/"features").iterdir() if p.is_dir()]
    return sorted(folds, key=lambda x:int(x.split("_")[-1]))

def _load_np_csv(root: Path, fold: str, base: str):
    feats = np.load(root/"features"/fold/f"{base}.npy")             # (N, D)
    sc = load(root/"features"/fold/"scaler.joblib")
    feats = sc.transform(feats).astype(np.float32)
    df = pd.read_csv(root/"splits"/fold/f"{base}.csv")
    # 与训练保持一致的情绪过滤
    if "emotion" in df.columns:
        keep = df["emotion"].isin(EMO_LABELS)
        feats = feats[keep.values]
        df = df[keep]
    return feats, df.reset_index(drop=True)

def _save_meta(df: pd.DataFrame, out_dir: Path, tag: str):
    keep_cols = [c for c in ["id","image","path","emotion","gender"] if c in df.columns]
    if "emotion" in keep_cols and "emotion" in df.columns:
        # 保留 remapped 以便 attacker 使用
        df = df.copy()
        df["emotion_remap"] = df["emotion"].map(EMO_LABELS).astype(int)
    df[keep_cols + (["emotion_remap"] if "emotion" in keep_cols else [])].to_csv(
        out_dir/f"{tag}_meta.csv", index=False
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="RAE root with features/ + splits/")
    ap.add_argument("--exp",  type=Path, required=True, help="experiment dir containing fold_* with checkpoints (we will save embeddings inside each fold)")
    ap.add_argument("--grl-layer", type=int, default=4, help="GRL tap layer used when training the model (rebuilds the same head dims)")
    ap.add_argument("--layers", default="L1,L2,L3,L4", help="comma list among L1,L2,L3,L4")
    ap.add_argument("--from-tags", default="risk_train,risk_val,test", help="comma list of source split basenames")
    ap.add_argument("--to-tags",   default="trans_train,trans_val,test", help="comma list of target embedding tags")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    layers = [s.strip().upper() for s in args.layers.split(",")]
    from_tags = [s.strip() for s in args.from_tags.split(",")]
    to_tags   = [s.strip() for s in args.to_tags.split(",")]
    assert len(from_tags)==len(to_tags), "--from-tags and --to-tags must align"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = _list_folds(args.root)
    print("Folds:", folds)

    for fold in folds:
        print(f"\n=== {fold} ===")
        fold_dir = args.exp/fold
        # 找 checkpoint（你之前保存的名字是 mlp_best_*.pt）
        ckpts = sorted(fold_dir.glob("*.pt"))
        assert len(ckpts)>0, f"No checkpoint found in {fold_dir}"
        ckpt = ckpts[0]
        # 构建与训练一致的 MLP
        # in_dim from features; n_emo=4（你当前 speech 只保留4类）
        # 用任意 split 先读出维度
        sample_X, _ = _load_np_csv(args.root, fold, from_tags[0])
        in_dim = sample_X.shape[1]
        model = MLP(in_dim=in_dim, n_emo=4, grl_layer=args.grl_layer).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        model.eval()
        out_emb = fold_dir/"embeddings"; out_emb.mkdir(parents=True, exist_ok=True)

        for src, dst in zip(from_tags, to_tags):
            X, meta = _load_np_csv(args.root, fold, src)
            ld = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=args.bs, shuffle=False)
            # 收集各层
            buf = {L:[] for L in layers}
            with torch.no_grad():
                for (xb,) in ld:
                    z = model.extract_layers(xb.to(device))
                    for L in layers:
                        buf[L].append(z[L].cpu().numpy())
            for L in layers:
                mat = np.concatenate(buf[L], axis=0)
                np.save(out_emb/f"{dst}_{L}.npy", mat)
            _save_meta(meta, out_emb, dst)
            print(f"[{dst}] saved:", ", ".join([f"{dst}_{L}.npy({len(buf[L])})" for L in layers]))

if __name__ == "__main__":
    main()
