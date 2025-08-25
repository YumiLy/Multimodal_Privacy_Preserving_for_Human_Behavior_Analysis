#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export attacker embeddings for face (RAF-DB).
Loads a trained MLP checkpoint and projects attacker_train / attacker_val
into L1/L2/L3/L4(Final) embeddings for later attacker training.
"""

import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from joblib import load
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dims=[2048,512,256,128], drop=0.3, n_emo=4, grl_layer=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_dim
        for h in hidden_dims:
            self.blocks.append(
                nn.Sequential(nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(drop))
            )
            prev = h
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_dims[grl_layer-1], 128), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(128, 2)
        )
        self.emotion_head = nn.Linear(hidden_dims[-1], n_emo)

    @torch.no_grad()
    def extract_layers(self, x):
        outs = {}
        h = x
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h)
            outs[f"L{i}"] = h
        outs["Final"] = outs["L4"]
        return outs

def load_feats_labels(split_dir: Path, tag: str, scaler):
    X = np.load(split_dir / f"{tag}_feats.npy")
    df = pd.read_csv(split_dir / f"{tag}_labels.csv")
    X = scaler.transform(X).astype(np.float32)
    return X, df

def save_embeddings(model, X, df, out_dir: Path, tag: str, bs=256, device="cpu"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ld = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs, shuffle=False)

    buf = {L: [] for L in ["L1", "L2", "L3", "L4", "Final"]}
    with torch.no_grad():
        for (xb,) in ld:
            feats = model.extract_layers(xb.to(device))
            for L in ["L1", "L2", "L3", "L4", "Final"]:
                buf[L].append(feats[L].cpu().numpy())

    for L in ["L1", "L2", "L3", "L4", "Final"]:
        mat = np.concatenate(buf[L], axis=0)
        np.save(out_dir / f"{tag}_{L}.npy", mat)

    # meta：尽量保留 path/image + emotion + gender
    keep = [c for c in ["image", "path", "emotion", "gender"] if c in df.columns]
    meta = df[keep].copy()
    meta.to_csv(out_dir / f"{tag}_meta.csv", index=False)

    shapes = {L: np.concatenate(buf[L], axis=0).shape for L in ["L1","L2","L3","L4","Final"]}
    print(f"[dump] {tag}: {shapes}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=Path, required=True,
                    help="Dir containing attacker_train_feats.npy / attacker_train_labels.csv etc.")
    ap.add_argument("--exp", type=Path, required=True,
                    help="Experiment dir with trained model checkpoint + scaler.joblib")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Checkpoint filename under --exp (e.g., mlp_best_L4_logistic_lam0.8_g10.0.pt)")
    ap.add_argument("--out_subdir", type=str, default="embeddings",
                    help="Subdir under --exp to save embeddings (default: embeddings)")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load scaler
    scaler_path = args.exp / "scaler.joblib"
    scaler = load(scaler_path)

    # 2) build model and load ckpt
    X_sample, _ = load_feats_labels(args.split_dir, "attacker_train", scaler)
    in_dim = X_sample.shape[1]
    model = MLP(in_dim=in_dim).to(device)
    ckpt_path = args.exp / args.ckpt
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    out_dir = args.exp / args.out_subdir

    # 3) export attacker_train / attacker_val
    for tag in ["attacker_train", "attacker_val"]:
        X, df = load_feats_labels(args.split_dir, tag, scaler)
        save_embeddings(model, X, df, out_dir, tag, bs=args.bs, device=device)

if __name__ == "__main__":
    main()
