#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_split.py  (with min-emotion-frac filter + rich reports)
------------------------------------------------------------
- 先在 *训练集* 上计算各 emotion 的占比，保留占比 >= --min-emotion-frac 的情绪
- 将同一组情绪应用到 test（保证标签空间一致）
- train 侧默认丢弃 unknown gender；test 侧通过 --keep-unknown-test 控制
- 分层依据：emotion × gender
- 输出：main_train / main_val / attacker_train / attacker_val / test
- 打印每个 split 的 emotion / gender / emotion×gender 分布

Usage:
  python data_split.py \
    --img-train /home/lai/FER/data/DATASET/train \
    --img-test  /home/lai/FER/data/DATASET/test  \
    --train-csv /home/lai/FER/data/train_labels_with_gender.csv \
    --test-csv  /home/lai/FER/data/test_labels_with_gender.csv  \
    --out-dir   /home/lai/FER/data/splits_features \
    --main-frac 0.70 --val-frac 0.15 --seed 42 \
    --min-emotion-frac 0.10 \
    --keep-unknown-test
"""
import argparse, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from collections import OrderedDict

EMO_ID2NAME = OrderedDict([
    (1, "Surprise"),
    (2, "Fear"),
    (3, "Disgust"),
    (4, "Happiness"),
    (5, "Sadness"),
    (6, "Anger"),
    (7, "Neutral"),
])

GENDER_ORDER = ["male", "female"]

def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def build_name_to_path(img_train: Path, img_test: Path):
    imgs = collect_images(img_train) + collect_images(img_test)
    return {p.name.lower(): str(p) for p in imgs}

def resolve_path(name_to_path, fname: str):
    return name_to_path.get(str(fname).strip().lower())

def keep_cols(dfx: pd.DataFrame) -> pd.DataFrame:
    # 标准列名：image, path, emotion, gender
    return dfx[["image", "path", "label", "perceived_gender"]]\
             .rename(columns={"label":"emotion", "perceived_gender":"gender"})

def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_te = next(sss.split(df, df["strata"]))
    return df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)

def report(df_out: pd.DataFrame, name: str):
    print(f"\n== {name} ==")
    print("TOTAL:", len(df_out))
    emo_counts = df_out["emotion"].value_counts().sort_index()
    gen_counts = df_out["gender"].value_counts().sort_index()
    print("emotion counts:\n", emo_counts.to_string())
    print("gender  counts:\n", gen_counts.to_string())
    ct = pd.crosstab(df_out["emotion"], df_out["gender"]).sort_index()
    print("emotion × gender:\n", ct.to_string())

def dist_report_and_plot_pretty(df, name: str, out_dir: Path, dpi=300):
    """
    - 打印 (emotion, gender) 交叉计数
    - 画柱状图：情绪用名字作为横轴；只画当前 split 里存在的情绪（不存在的直接不画）
    - 在柱子上方标百分比
    - 保存为 <name>_distribution.png
    """
    # 统一列名（你的 keep_cols 已经做过；这里再保险）
    df = df.rename(columns={"label":"emotion", "perceived_gender":"gender"})

    # ---- 文本统计 ----
    print(f"\n== {name} ==")
    cross = pd.crosstab(df["emotion"], df["gender"])
    print(cross.to_string())

    # ---- Emotion 统计（只保留当前 split 出现过的情绪，并映射名字）----
    present_ids = sorted(set(df["emotion"].astype(int).unique().tolist()))
    # 只画存在的；并按官方顺序筛选子集
    emo_ids_plot = [eid for eid in EMO_ID2NAME.keys() if eid in present_ids]
    emo_names    = [EMO_ID2NAME[eid] for eid in emo_ids_plot]

    emo_counts_full = df["emotion"].astype(int).value_counts()
    emo_counts = [int(emo_counts_full.get(eid, 0)) for eid in emo_ids_plot]
    emo_total  = max(sum(emo_counts), 1)
    emo_fracs  = [c / emo_total for c in emo_counts]

    # ---- Gender 统计（同样只画出现过的）----
    gen_counts_full = df["gender"].astype(str).str.lower().value_counts()
    gen_keys_plot   = [g for g in GENDER_ORDER if g in gen_counts_full.index]
    gen_counts      = [int(gen_counts_full[g]) for g in gen_keys_plot]
    gen_total       = max(sum(gen_counts), 1)
    gen_fracs       = [c / gen_total for c in gen_counts]

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Emotion
    axes[0].bar(range(len(emo_names)), emo_counts, color="skyblue", edgecolor="black")
    axes[0].set_title("Expression")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(len(emo_names)))
    axes[0].set_xticklabels(emo_names, rotation=45, ha="right")
    for i, (c, f) in enumerate(zip(emo_counts, emo_fracs)):
        axes[0].text(i, c + max(1, 0.01 * emo_total), f"{100*f:.2f}%", ha="center", va="bottom", fontsize=8)

    # Gender
    axes[1].bar(range(len(gen_keys_plot)), gen_counts, color="skyblue", edgecolor="black")
    axes[1].set_title("Gender")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(range(len(gen_keys_plot)))
    axes[1].set_xticklabels([g.title() for g in gen_keys_plot], rotation=45, ha="right")
    for i, (c, f) in enumerate(zip(gen_counts, gen_fracs)):
        axes[1].text(i, c + max(1, 0.01 * gen_total), f"{100*f:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle(name)
    fig.tight_layout()
    out_path = out_dir / f"{name}_distribution.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-train", type=Path, required=True)
    ap.add_argument("--img-test",  type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--test-csv",  type=Path, required=True)
    ap.add_argument("--out-dir",   type=Path, required=True)
    ap.add_argument("--main-frac", type=float, default=0.70, help="portion of official train going to main task")
    ap.add_argument("--val-frac",  type=float, default=0.15, help="within each bucket, portion for validation")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--drop-unknown-train", action="store_true", default=True,
                    help="drop unknown genders from train (default True)")
    ap.add_argument("--keep-unknown-test", action="store_true",
                    help="keep unknown genders in test; otherwise drop them")
    ap.add_argument("--min-emotion-frac", type=float, default=0.10,
                    help="keep emotions whose TRAIN share >= this fraction (e.g., 0.10)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读入 + 路径映射
    print("[info] loading CSVs & mapping paths...")
    df_tr = pd.read_csv(args.train_csv)
    df_te = pd.read_csv(args.test_csv)
    required = {"image", "label", "perceived_gender"}
    for tag, df in [("train", df_tr), ("test", df_te)]:
        miss = required - set(df.columns)
        if miss: raise ValueError(f"{tag} CSV missing columns: {miss}")

    name_to_path = build_name_to_path(args.img_train, args.img_test)
    df_tr["path"] = df_tr["image"].map(lambda x: resolve_path(name_to_path, x))
    df_te["path"] = df_te["image"].map(lambda x: resolve_path(name_to_path, x))
    miss_tr = df_tr["path"].isna().sum(); miss_te = df_te["path"].isna().sum()
    print(f"[path map] train missing={miss_tr}/{len(df_tr)}  test missing={miss_te}/{len(df_te)}")
    if miss_tr or miss_te:
        print("[warn] dropping rows with missing file paths...")
        df_tr = df_tr.dropna(subset=["path"]).reset_index(drop=True)
        df_te = df_te.dropna(subset=["path"]).reset_index(drop=True)

    # 2) 训练侧去 unknown gender；测试端按需
    if args.drop_unknown_train:
        before = len(df_tr)
        df_tr = df_tr[df_tr["perceived_gender"].isin(["male","female"])].copy()
        print(f"[filter train] kept known genders: {len(df_tr)}/{before}")
    if not args.keep_unknown_test:
        before = len(df_te)
        df_te = df_te[df_te["perceived_gender"].isin(["male","female"])].copy()
        print(f"[filter test]  kept known genders: {len(df_te)}/{before}")
    else:
        print("[test] keeping unknown genders as-is.")

    # 3) 只保留占比 >= min-emotion-frac 的情绪（基于训练集分布）
    print(f"[filter emotion] min-emotion-frac = {args.min_emotion_frac:.2f} (computed on TRAIN)")
    emo_frac = df_tr["label"].value_counts(normalize=True).sort_index()
    print("train emotion fraction:")
    print((emo_frac*100).round(2).astype(str) + "%")
    keep_emos = set(emo_frac[emo_frac >= args.min_emotion_frac].index.tolist())
    print("keep emotions:", sorted(keep_emos))
    if len(keep_emos) < 2:
        raise ValueError("Too few emotions kept. Lower --min-emotion-frac or check your data.")

    before_tr, before_te = len(df_tr), len(df_te)
    df_tr = df_tr[df_tr["label"].isin(keep_emos)].copy()
    df_te = df_te[df_te["label"].isin(keep_emos)].copy()
    print(f"[filter emotion] train kept {len(df_tr)}/{before_tr} | test kept {len(df_te)}/{before_te}")

    # 4) 分层键
    df_tr["strata"] = df_tr["label"].astype(str) + "_" + df_tr["perceived_gender"].astype(str)

    # 5) 外层：MAIN vs ATTACKER
    main_frac = float(args.main_frac)
    if not (0 < main_frac < 1):
        raise ValueError("--main-frac must be in (0,1)")
    df_main_all, df_att_all = stratified_split(df_tr, test_size=1.0 - main_frac, seed=args.seed)

    # 6) 内层：各自 train/val
    val_frac = float(args.val_frac)
    if not (0 < val_frac < 1):
        raise ValueError("--val-frac must be in (0,1)")
    for d in (df_main_all, df_att_all):
        d["strata"] = d["label"].astype(str) + "_" + d["perceived_gender"].astype(str)
    df_main_train, df_main_val = stratified_split(df_main_all, test_size=val_frac, seed=args.seed+1)
    df_att_train,  df_att_val  = stratified_split(df_att_all,  test_size=val_frac, seed=args.seed+2)

    # 7) 保存
    main_train_csv = args.out_dir / "main_train.csv"
    main_val_csv   = args.out_dir / "main_val.csv"
    att_train_csv  = args.out_dir / "attacker_train.csv"
    att_val_csv    = args.out_dir / "attacker_val.csv"
    test_csv_out   = args.out_dir / "test.csv"

    keep_cols(df_main_train).to_csv(main_train_csv, index=False)
    keep_cols(df_main_val).to_csv(main_val_csv,   index=False)
    keep_cols(df_att_train).to_csv(att_train_csv, index=False)
    keep_cols(df_att_val).to_csv(att_val_csv,     index=False)
    keep_cols(df_te).to_csv(test_csv_out, index=False)

    print("\n[saved]")
    for p in [main_train_csv, main_val_csv, att_train_csv, att_val_csv, test_csv_out]:
        print(" -", p, f"({os.path.getsize(p)} bytes)")

    # 8) 分布报告
    report(keep_cols(df_main_train), "main_train")
    report(keep_cols(df_main_val),   "main_val")
    report(keep_cols(df_att_train),  "attacker_train")
    report(keep_cols(df_att_val),    "attacker_val")
    report(keep_cols(df_te),         "test")

    dist_report_and_plot_pretty(keep_cols(df_main_train), "main_train", args.out_dir)
    dist_report_and_plot_pretty(keep_cols(df_main_val),   "main_val",   args.out_dir)
    dist_report_and_plot_pretty(keep_cols(df_att_train),  "attacker_train", args.out_dir)
    dist_report_and_plot_pretty(keep_cols(df_att_val),    "attacker_val",   args.out_dir)
    dist_report_and_plot_pretty(keep_cols(df_te),         "test",       args.out_dir)



if __name__ == "__main__":
    main()
