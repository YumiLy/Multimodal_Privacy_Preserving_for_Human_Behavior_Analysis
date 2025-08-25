#!/usr/bin/env python3
"""data_prep.py

Extracts the **Interspeech 2013 ComParE Challenge** acoustic feature set for a
list of input audio files, performs min‑max normalisation and dimensionality
reduction with PCA (6373 → 256), then saves the processed features along with
the fitted scaler/PCA objects for later inference.

Usage (example):
    python data_prep.py \
        --filelist train_list.txt test_list.txt \
        --root_audio /path/to/wavs \
        --out_dir features/

train_list.txt / test_list.txt
    • Plain‑text files, **one relative wav path per line** (no header).
      The first filelist is treated as the *training* set – scaler & PCA are
      fitted only on its statistics; the remaining lists are transformed with
      the trained models.

Outputs (PCA 下的维度)
    ├── features/
    │   ├── scaler.joblib   # fitted sklearn MinMaxScaler
    │   ├── pca.joblib      # fitted sklearn PCA (256 comps)
    │   ├── train.npy       # N×256 float32
    │   ├── test.npy        # M×256 float32
    │   └── <more>.npy      # if you provided more filelists

Requirements
    pip install opensmile==2.4.1 scikit‑learn joblib soundfile tqdm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
import pandas as pd

import joblib
import numpy as np
import opensmile
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_filelist(path: Path, col: str = "wav_file") -> List[Path]:
    """
    Accept either:
        • A plain‑text (one path per line) file with relative wav paths, **or**
        • A CSV file with a column named `col` containing relative wav paths.
    Returns a list of relative paths (as Path objects).
    """
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"CSV file {path} does not contain column '{col}'")
    wav_paths = df[col].astype(str).tolist()
    return [Path(p) for p in wav_paths if isinstance(p, str) and p.strip()]
    


def extract_features(smile: opensmile.Smile, wav_path: Path) -> np.ndarray:
    """Compute ComParE 2013 features for one wav file and return 1‑D array."""
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {wav_path}: {e}") from e

    if audio.ndim > 1:  # convert stereo to mono
        audio = audio.mean(axis=1)

    feats_df = smile.process_signal(audio, sr)
    # feats_df is a pandas DataFrame with a single row (functionals)
    return feats_df.iloc[0].to_numpy(dtype=np.float32)


# 
def resolve_path(token: str, root: Path) -> Path:
    if token.startswith("Ses"):
        utterance_folder = '_'.join(token.split('_')[:-1]) 
        session_num = str(int(token[3:5]))
        return (root / "IEMOCAP/wav"
                / f"Session{session_num}" / "sentences/wav"
                / utterance_folder / f"{token}.wav")
    if token.startswith("Actor_"):
        return root / "RAVDESS" / token
    if token[:4].isdigit():
        fname = token if token.lower().endswith('.wav') else f"{token}.wav"
        return root / "CREMA-D/AudioWAV" / fname

    raise FileNotFoundError(token)



# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ComParE‑2016 feature extraction + min‑max scaler + PCA (6373(?)→256)")
    parser.add_argument('--filelist', required=True, nargs='+', type=Path,
                        help='One or more text files; first is training set.')
    parser.add_argument('--root_audio', type=Path, default=Path('.'),
                        help='Root dir prepended to each wav path in filelists.')
    parser.add_argument('--out_dir', type=Path, required=True,
                        help='Directory to save numpy features & models.')
    # parser.add_argument('--n_components', type=int, default=256, help='PCA components.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Dataset:", args.out_dir.name)



    # 1. Prepare OpenSMILE extractor (ComParE 2013, functional-level)
    print("Using OpenSMILE version:", opensmile.__version__)
    smile = opensmile.Smile(
        feature_set  = opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("Feature set:", smile.feature_set)
    print("Loading OpenSMILE model finished")

    all_sets = []  # list of (name, features array)

    print("\n[1/3] Extracting ComParE‑2013 features… ")

    for idx, list_path in enumerate(args.filelist):
        name = list_path.stem  # e.g. train / dev / test
        # wav_rel_paths = read_filelist(list_path)
        # feats = []
        # root = Path(args.root_audio)
        # for rel_path in tqdm(wav_rel_paths, desc=f"{name}"):
        #     wav_path = resolve_path(str(rel_path), root)
        #     # print("DEBUG:", rel_path, "→", wav_path) 
        #     feats.append(extract_features(smile, wav_path))
        # feats = np.stack(feats)
        # all_sets.append((name, feats))
        # print(f"  ✓ {name}: {feats.shape[0]} files × {feats.shape[1]} dims")
        df_meta = pd.read_csv(list_path)  # 必须含 wav_file
        wav_rel_paths = read_filelist(list_path)

        feats = []
        root = Path(args.root_audio)
        for rel_path in tqdm(wav_rel_paths, desc=name):
            wav_path = resolve_path(str(rel_path), root)
            feats.append(extract_features(smile, wav_path))

        feats = np.stack(feats).astype(np.float32)

        if len(df_meta) != len(feats):
            raise RuntimeError(f"{name}: CSV 行数 {len(df_meta)} 与特征 {len(feats)} 不符")

        # 保存对齐索引（原标签）
        # df_meta.to_csv(args.out_dir / f"{name}_index.csv", index=False)

        all_sets.append((name, feats, df_meta))
        print(f"{name}: {feats.shape}")

    # 2. Fit MinMaxScaler on training set only
    risk_train_feats = all_sets[0][1] # risk_train.npy
    trans_train_feats = all_sets[2][1] # trans_train.npy
    train_feats = np.concatenate([risk_train_feats, trans_train_feats])
    # print("\n[2/3] Fitting MinMaxScaler on training set…")
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_feats)
    # 这里改为standard scaler
    print("\n[2/3] Fitting StandardScaler on training set…")
    scaler = StandardScaler().fit(train_feats)

    # 3. Fit PCA on *scaled* training set
    # print("[3/4] Fitting PCA ({} → {} components) on training set…".format(train_feats.shape[1], args.n_components))
    # pca = PCA(n_components=args.n_components, random_state=args.seed).fit(scaler.transform(train_feats))

    # 4. Transform all sets & save
    print("[3/3] Saving…")
    for name, feats, _meta in all_sets:
        # feats_scaled = scaler.transform(feats)
        # feats_pca    = pca.transform(feats_scaled).astype(np.float32)
        # np.save(args.out_dir / f"{name}.npy", feats_pca)
        # print(f"  → {name}.npy: {feats_pca.shape}")
        np.save(args.out_dir / f"{name}.npy", feats)
        print(f"  → {name}.npy: {feats.shape}")

    # Persist scaler & PCA for inference
    joblib.dump(scaler, args.out_dir / 'scaler.joblib')
    # joblib.dump(pca,    args.out_dir / 'pca.joblib')
    print("\n✓ All done. Models saved to", args.out_dir)


    print("-------------------------------------------------------------------------------------")


if __name__ == '__main__':
    # Make sure script dir is in PYTHONPATH for opensmile models on some setups
    sys.path.append(str(Path(__file__).resolve().parent))
    main()
