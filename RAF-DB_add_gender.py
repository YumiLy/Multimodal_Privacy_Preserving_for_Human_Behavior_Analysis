
import os

# —— 限制到单线程，避免 ORT 设亲和性 ——
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# —— 关闭/放松线程亲和性（Intel/LLVM OpenMP）——
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"  # 可选

# —— 降低 onnxruntime 日志等级（可选）——
os.environ["ORT_LOG_severity_level"] = "3"



from pathlib import Path
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
import cv2, pandas as pd
from insightface.app import FaceAnalysis

# ==== PATHS (edit if needed) ====
# Image root where your RAF-DB images are stored (change to 'train' if you want)
IMG_ROOT_train = Path("/home/lai/FER/data/DATASET/train")
IMG_ROOT_test = Path("/home/lai/FER/data/DATASET/test")

# Existing labels CSV you want to augment with gender
CSV_IN_train = Path("/home/lai/FER/data/train_labels.csv")
CSV_IN_test = Path("/home/lai/FER/data/test_labels.csv")

# Output CSV path
CSV_OUT_train = CSV_IN_train.with_name(CSV_IN_train.stem + "_with_gender.csv")
CSV_OUT_test = CSV_IN_test.with_name(CSV_IN_test.stem + "_with_gender.csv")

# Inference settings
DETECTOR = "retinaface"   # 'retinaface' is strong; alternatives: 'opencv', 'mtcnn', 'mediapipe', 'yolov8'
THRESHOLD = 0.70          # probability threshold; below this => 'unknown'
BATCH_PRINT = 200         # progress print every N rows


import os

def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            out.append(p)
    return out

def basename(p: Path):
    return p.name

BACKENDS_TRY = ["mediapipe", "retinaface", "opencv", "mtcnn", "yolov8"]  
def analyze_gender(path: str):
    try:
        img = cv2.imread(path)
        if img is None:
            return "unknown", 0.0
        # RAF-DB aligned 有时很小，先粗暴放大一倍再试
        if min(img.shape[:2]) < 80:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        for backend in BACKENDS_TRY:
            try:
                res = DeepFace.analyze(
                    img_path = img,           # 直接传 ndarray，省 IO
                    actions = ["gender"],
                    detector_backend = backend,
                    enforce_detection = True,
                    prog_bar = False
                )
                obj = res[0] if isinstance(res, list) else res
                gd = obj.get("gender", {})
                man = float(gd.get("Man", 0))/100.0
                wom = float(gd.get("Woman", 0))/100.0
                conf = max(man, wom)
                if conf < THRESHOLD:
                    return "unknown", conf
                return ("male", conf) if man >= wom else ("female", conf)
            except Exception:
                continue
        return "unknown", 0.0
    except Exception:
        return "unknown", 0.0



app = FaceAnalysis(name="buffalo_l"); app.prepare(ctx_id=0, det_size=(640,640))
def analyze_gender_insightface(path):
    img = cv2.imread(path)
    if img is None: return "unknown", 0.0
    faces = app.get(img)
    if not faces:    return "unknown", 0.0
    f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    label = "male" if int(getattr(f, "gender", -1))==1 else "female"
    return label, 0.70  # 没有显式概率，给一个保守置信度占位


THRESHOLD = 0.70

def analyze_gender_strong(path: str, log_every=None):
    # 1) 先用和你单张测试完全一致的调用：
    try:
        res = DeepFace.analyze(
            img_path=path,                 # 传“路径”，不要传 ndarray
            actions=["gender"],
            detector_backend="retinaface", # 和你单张一致
            enforce_detection=True
        )
        obj = res[0] if isinstance(res, list) else res
        gd  = obj.get("gender", {})
        man = float(gd.get("Man", 0))/100.0
        wom = float(gd.get("Woman", 0))/100.0
        conf = max(man, wom)
        return (("male", conf) if man >= wom else ("female", conf)) if conf >= THRESHOLD else ("unknown", conf)
    except Exception as e:
        if log_every:
            print(f"[W] retinaface failed: {path} ({e})")

    # 2) 退而求其次：跳过检测（对已对齐的脸很有用）
    try:
        res = DeepFace.analyze(
            img_path=path,                 # 仍然传“路径”
            actions=["gender"],
            detector_backend="skip",
            enforce_detection=False
        )
        obj = res[0] if isinstance(res, list) else res
        gd  = obj.get("gender", {})
        man = float(gd.get("Man", 0))/100.0
        wom = float(gd.get("Woman", 0))/100.0
        conf = max(man, wom)
        return (("male", conf) if man >= wom else ("female", conf)) if conf >= THRESHOLD else ("unknown", conf)
    except Exception as e:
        if log_every:
            print(f"[W] skip failed: {path} ({e})")

    # 3) 最后兜底：InsightFace
    try:
        img = cv2.imread(path)            # InsightFace 接受 ndarray（BGR）
        if img is None:
            if log_every: print(f"[E] cv2.imread failed: {path}")
            return "unknown", 0.0
        faces = app.get(img)
        if not faces:
            return "unknown", 0.0
        f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        label = "male" if int(getattr(f, "gender", -1)) == 1 else "female"
        return label, 0.70
    except Exception as e:
        if log_every:
            print(f"[E] InsightFace failed: {path} ({e})")
        return "unknown", 0.0


# 1) Read your existing CSV and ensure we have an 'image' column matching basenames
df_train = pd.read_csv(CSV_IN_train)
if "image" not in df_train.columns:
    raise ValueError(f"'image' column not found in {CSV_IN_train}. Please ensure the CSV has headers: image,label")
df_test = pd.read_csv(CSV_IN_test)
if "image" not in df_test.columns:
    raise ValueError(f"'image' column not found in {CSV_IN_test}. Please ensure the CSV has headers: image,label")

# 2) Map from basename -> full path (we'll scan the folder recursively)
imgs_train = collect_images(IMG_ROOT_train)
imgs_test  = collect_images(IMG_ROOT_test)
imgs = imgs_train + imgs_test

name_to_path = {p.name.lower(): str(p) for p in imgs}


def resolve_path(fname: str):
    return name_to_path.get(str(fname).strip().lower())

# 自检：看看映射成功率
import os
def sanity_check_paths(df, tag):
    missing = sum(resolve_path(x) is None for x in df["image"])
    print(f"[{tag}] total={len(df)}, missing={missing} ({missing/len(df):.2%})")
sanity_check_paths(df_train, "train")
sanity_check_paths(df_test,  "test")

p = resolve_path(df_test.iloc[0]["image"])
print("sample:", p)
print(analyze_gender_strong(p, log_every=True))


# # 3) For rows in CSV, run gender analysis if file exists; else mark unknown

# gender_labels = []
# gender_conf = []

# for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
#     fname = str(row["image"])
#     fpath = name_to_path.get(fname)
#     if fpath is None:
#         # Try to be tolerant: if CSV has only the basename, maybe the image lives under another subfolder
#         # We'll keep 'unknown' if not found
#         gender_labels.append("unknown")
#         gender_conf.append(0.0)
#         if (i+1) % BATCH_PRINT == 0:
#             print(f"[{i+1}/{len(df_train)}] file not found: {fname}")
#         continue

#     label, conf = analyze_gender(fpath)
#     gender_labels.append(label)
#     gender_conf.append(conf)

#     if (i+1) % BATCH_PRINT == 0:
#         print(f"[{i+1}/{len(df_train)}] last: {fname} -> {label} ({conf:.2f})")

# df_train["perceived_gender"] = gender_labels
# df_train["gender_confidence"] = gender_conf

# # 4) Save
# df_train.to_csv(CSV_OUT_train, index=False)
# CSV_OUT_train




# gender_labels = []
# gender_conf = []

# for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
#     fname = str(row["image"])
#     fpath = name_to_path.get(fname)
#     if fpath is None:
#         # Try to be tolerant: if CSV has only the basename, maybe the image lives under another subfolder
#         # We'll keep 'unknown' if not found
#         gender_labels.append("unknown")
#         gender_conf.append(0.0)
#         if (i+1) % BATCH_PRINT == 0:
#             print(f"[{i+1}/{len(df_test)}] file not found: {fname}")
#         continue

#     label, conf = analyze_gender(fpath)
#     gender_labels.append(label)
#     gender_conf.append(conf)

#     if (i+1) % BATCH_PRINT == 0:
#         print(f"[{i+1}/{len(df_test)}] last: {fname} -> {label} ({conf:.2f})")

# df_test["perceived_gender"] = gender_labels
# df_test["gender_confidence"] = gender_conf

# # 4) Save
# df_test.to_csv(CSV_OUT_test, index=False)
# CSV_OUT_test

# ---------- 批处理（train / test 通用函数） ----------
def label_csv(df, tag):
    gender_labels, gender_conf = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{tag}"):
        fname = str(row["image"])
        fpath = resolve_path(fname)
        if fpath is None:
            # 文件名没匹配上（最常见问题）
            gender_labels.append("unknown"); gender_conf.append(0.0)
            if (i+1) % BATCH_PRINT == 0:
                print(f"[{tag} {i+1}] file not found: {fname}")
            continue

        label, conf = analyze_gender_strong(fpath, log_every=((i+1) % BATCH_PRINT == 0))
        gender_labels.append(label); gender_conf.append(conf)

        if (i+1) % BATCH_PRINT == 0:
            print(f"[{tag} {i+1}] last: {fname} -> {label} ({conf:.2f})")

    df["perceived_gender"] = gender_labels
    df["gender_confidence"] = gender_conf
    return df

df_train = label_csv(df_train, "train")
df_test  = label_csv(df_test,  "test")

df_train.to_csv(CSV_OUT_train, index=False)
df_test.to_csv(CSV_OUT_test, index=False)
print("Saved:", CSV_OUT_train, CSV_OUT_test)
