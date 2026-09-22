# -*- coding: utf-8 -*-
"""
LWST-HGN: A Learnable Wavelet-Enhanced Spatio-Temporal Hypergraph Network for Soft Sensing
for soft sensing on the UCI Air Quality dataset (C6H6(GT) task).

This open-source training script is the cleaned release version corresponding to
our best AirQuality/C6H6_GT configuration. The learning/model logic is preserved
from the experiment code; only public-facing names, comments, paths, and default
arguments are cleaned for reproducibility.

Default best configuration:
- target: C6H6_GT
- chronological split: 60/20/20 (segment mode)
- window length: 3
- training-set Pearson screening: remove the 3 least-correlated inputs
- static hypergraph, k=3
- MS-GTC kernels: {3,5,7}, gated fusion
- learnable wavelet initialized by db2
- wavelet residual scale alpha=0.055, low-frequency ratio rho=0.5
- energy regularization weight: 1e-4
- Huber loss, lr=1.5e-3, weight decay=1e-4, batch size=128, dropout=0.1
- seed=3407

Quick start:
    python train_airquality_c6h6.py --data_path data/AirQualityUCI.csv --gpu 0
"""

import os
import re
import json
import copy
import glob
import argparse
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utilities
# =========================================================

# 原始列名保留 UCI 命名，内部统一成更适合命令行的列名
RAW_TO_CANONICAL = {
    "Date": "Date",
    "Time": "Time",
    "CO(GT)": "CO_GT",
    "PT08.S1(CO)": "PT08_S1_CO",
    "NMHC(GT)": "NMHC_GT",
    "C6H6(GT)": "C6H6_GT",
    "PT08.S2(NMHC)": "PT08_S2_NMHC",
    "NOx(GT)": "NOx_GT",
    "PT08.S3(NOx)": "PT08_S3_NOx",
    "NO2(GT)": "NO2_GT",
    "PT08.S4(NO2)": "PT08_S4_NO2",
    "PT08.S5(O3)": "PT08_S5_O3",
    "T": "T",
    "RH": "RH",
    "AH": "AH",
}

REQUIRED_COLS = [
    "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC",
    "NOx_GT", "PT08_S3_NOx", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3",
    "T", "RH", "AH",
]

# 严格软测量协议：只用可在线获得的金属氧化物传感器响应和气象变量，不使用其他 GT 分析仪浓度作为输入。
STRICT_INPUT_COLS = [
    "PT08_S1_CO", "PT08_S2_NMHC", "PT08_S3_NOx", "PT08_S4_NO2", "PT08_S5_O3",
    "T", "RH", "AH",
]

# 如果为了冲更好结果，可选 wide 协议：加入其他 GT 浓度作为辅助输入，但不能加入当前目标列。
AUX_GT_COLS = ["CO_GT", "C6H6_GT", "NOx_GT", "NO2_GT"]
TARGET_COLS = ["NO2_GT", "NOx_GT"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().numpy().tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    data_path: str = "./data/AirQualityUCI.csv"
    out_dir: str = "./outputs"

    target: str = "C6H6_GT"  # released best task
    input_cols: Optional[List[str]] = None
    input_mode: str = "strict"  # strict | wide

    # correlation screening module
    use_corr_filter: bool = True
    corr_remove_n: int = 3
    corr_remove_list: str = "3"  # released best setting

    # data protocol
    split_mode: str = "segment"  # segment | global | segment_blocked；单文件时 segment=连续60/20/20
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
    window: int = 3
    window_list: str = "3"  # released best setting
    stride: int = 1
    block_size: int = 168  # segment_blocked 模式下 val/test 交替块长度；小时数据可用一周
    fill_missing: str = "interpolate"  # 输入变量缺失处理：interpolate | drop
    target_fill: str = "drop"          # 目标缺失处理：drop | interpolate；默认不要插值标签
    target_transform: str = "log1p"    # released best setting

    # training
    batch_size: int = 128
    max_epochs: int = 150
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    dropout: float = 0.10
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-7
    seed: int = 3407
    loss: str = "huber"  # mse | huber | mixed_relative | relative_huber
    huber_delta: float = 1.0
    relative_loss_alpha: float = 0.3
    relative_loss_denom: float = 20.0

    # model selection / early stopping
    selection_metric: str = "nrmse"    # released best setting

    # model
    num_mixers: int = 2
    mixer_hidden: int = 64
    embed_dim: int = 32
    num_blocks: int = 2
    readout_hidden: int = 64
    kernel_size: int = 5
    dilation: int = 1

    # multiscale temporal fusion
    use_multiscale_gtc: bool = True
    multiscale_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    multiscale_dilations: Tuple[int, ...] = (1,)
    multiscale_fusion: str = "gated"  # conv | gated

    # wavelet residual enhancement
    # Fixed mode: use a fixed classical high-pass detail filter.
    # Learnable QMF mode: initialize a low-pass prototype from a classical wavelet,
    # derive the high-pass filter by the QMF relation, and optimize the low-pass prototype by backpropagation.
    # Energy regularization is used to stabilize learnable wavelet filter training.
    use_wavelet_enhance: bool = True
    wavelet_type: str = "db2"           # released best initialization
    wavelet_alpha: float = 0.055         # residual enhancement scale
    wavelet_position: str = "post_mixer" # pre_mixer | post_mixer
    wavelet_learnable: bool = True       # learnable low-pass prototype
    wavelet_learnable_alpha: bool = False # 0: fixed alpha; 1: learn alpha by sigmoid(logit)*wavelet_alpha_max
    wavelet_alpha_max: float = 0.50      # upper bound for learnable alpha
    wavelet_filter_norm: bool = True     # normalize learnable filter energy during forward
    wavelet_learnable_lowpass: bool = True  # enable low-frequency residual branch
    wavelet_lowpass_mode: str = "residual" # residual: use cA-x; direct: use cA
    wavelet_lowpass_alpha_ratio: float = 0.5 # low-frequency residual ratio rho
    wavelet_energy_lambda: float = 1e-4  # released best setting

    # hypergraph
    hypergraph_mode: str = "static"  # static | dynamic
    auto_k_search: bool = True
    k_candidates: Tuple[int, ...] = (3,)
    fixed_k: int = 3

    # runtime
    gpu: int = 0
    num_workers: int = 0
    save_predictions: bool = True
    save_adjacency: bool = False
    save_model: bool = False
    verbose: bool = True
    print_baseline: bool = False
    torch_num_threads: int = 4

    # metrics
    mape_zero_threshold: float = 1e-6
    mape_eps_denom: float = 1e-6

# =========================================================
# Data loading
# =========================================================

def _drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    # 去掉由 csv 尾部 ;; 产生的空列
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    return df.reset_index(drop=True)


def _excel_datetime_from_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    # Excel 日期序列：1899-12-30 为 pandas 默认 origin
    d = pd.to_datetime(pd.to_numeric(date_s, errors="coerce"), unit="D", origin="1899-12-30", errors="coerce")
    t_num = pd.to_numeric(time_s, errors="coerce")
    return d + pd.to_timedelta(t_num.fillna(0.0), unit="D")


def _parse_airquality_datetime(df: pd.DataFrame) -> pd.Series:
    date_s = df["Date"]
    time_s = df["Time"]
    # xlsx 中 Date/Time 通常是数字；csv 中是 10/03/2004 + 18.00.00
    if pd.api.types.is_numeric_dtype(date_s) or pd.api.types.is_numeric_dtype(time_s):
        return _excel_datetime_from_date_time(date_s, time_s)
    text = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip().str.replace(".", ":", regex=False)
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: RAW_TO_CANONICAL[c] for c in df.columns if c in RAW_TO_CANONICAL}
    df = df.rename(columns=rename)
    return df


def load_airquality_file(cfg: Config) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    path = cfg.data_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data file: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        # UCI 原始 csv 是分号分隔、逗号小数，末尾还有两个空列
        df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("data_path must be .csv/.txt/.xlsx/.xls")

    df = _drop_empty_rows_cols(df)
    df = _canonicalize_columns(df)

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError(f"Missing Date/Time columns. Existing columns={df.columns.tolist()}")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Existing columns={df.columns.tolist()}")

    df["DateTime"] = _parse_airquality_datetime(df)
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    # 转成数值，并把 -200 置为 NaN
    for c in REQUIRED_COLS:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == -200, c] = np.nan

    raw_missing = {c: int(df[c].isna().sum()) for c in REQUIRED_COLS}
    raw_missing_rate = {c: float(df[c].isna().mean()) for c in REQUIRED_COLS}

    # 不在这里插值目标列；输入列会在 prepare_target_data 里按实际任务插值。
    df = df[["DateTime"] + REQUIRED_COLS].copy()
    df["__segment_id__"] = 0
    df["__local_time__"] = np.arange(len(df), dtype=np.int64)
    df["__source_file__"] = os.path.basename(path)

    meta = {
        "path": path,
        "file": os.path.basename(path),
        "num_rows": int(len(df)),
        "datetime_start": str(df["DateTime"].iloc[0]) if len(df) else None,
        "datetime_end": str(df["DateTime"].iloc[-1]) if len(df) else None,
        "columns": REQUIRED_COLS,
        "raw_missing_count": raw_missing,
        "raw_missing_rate": raw_missing_rate,
        "min": {c: (None if df[c].dropna().empty else float(df[c].min())) for c in REQUIRED_COLS},
        "max": {c: (None if df[c].dropna().empty else float(df[c].max())) for c in REQUIRED_COLS},
        "mean": {c: (None if df[c].dropna().empty else float(df[c].mean())) for c in REQUIRED_COLS},
    }
    return [df], meta


def get_input_columns(cfg: Config, target_col: str) -> List[str]:
    if cfg.input_cols:
        cols: List[str] = []
        for x in cfg.input_cols:
            if "," in x:
                cols.extend([t.strip() for t in x.split(",") if t.strip()])
            else:
                cols.append(x.strip())
        return cols

    if cfg.input_mode == "strict":
        return [c for c in STRICT_INPUT_COLS if c != target_col]
    if cfg.input_mode == "wide":
        # 加入其他 GT 浓度作为辅助输入，但不能加入当前目标；NMHC_GT 缺失约90%，默认不加入。
        return [c for c in STRICT_INPUT_COLS + AUX_GT_COLS if c != target_col and c != "NMHC_GT"]
    raise ValueError("input_mode must be strict or wide")


# =========================================================
# Pearson correlation screening
# =========================================================

def _train_rows_for_corr(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> pd.DataFrame:
    """
    Only use training portion to compute correlation scores.
    This prevents validation/test leakage when selecting variables.
    """
    cols = list(dict.fromkeys(input_cols + [target_col]))
    if cfg.split_mode == "global":
        df_all = pd.concat(segments, axis=0, ignore_index=True)
        train_end = int(len(df_all) * cfg.split_train)
        return df_all.iloc[:train_end][cols].copy()

    # segment and segment_blocked both use each segment's first split_train portion for training.
    train_parts = []
    for df in segments:
        train_end = int(len(df) * cfg.split_train)
        train_parts.append(df.iloc[:train_end][cols].copy())
    return pd.concat(train_parts, axis=0, ignore_index=True)


def compute_train_abs_pearson_scores(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Dict[str, float]:
    """
    Compute |Pearson(input, target)| on training rows only.
    Rows with missing target are dropped. Inputs should already be interpolated by preprocess_missing_for_target().
    """
    df_train = _train_rows_for_corr(segments, input_cols, target_col, cfg)
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna(subset=[target_col])

    scores: Dict[str, float] = {}
    y = df_train[target_col].to_numpy(dtype=np.float64)
    y_ok = np.isfinite(y)
    for col in input_cols:
        x = df_train[col].to_numpy(dtype=np.float64)
        ok = y_ok & np.isfinite(x)
        if ok.sum() < 3:
            scores[col] = 0.0
            continue
        x0 = x[ok]
        y0 = y[ok]
        if np.std(x0) < 1e-12 or np.std(y0) < 1e-12:
            scores[col] = 0.0
            continue
        corr = float(np.corrcoef(x0, y0)[0, 1])
        scores[col] = 0.0 if not np.isfinite(corr) else abs(corr)
    return scores


def apply_correlation_screening(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Tuple[List[str], Dict[str, Any]]:
    remove_n = int(max(0, cfg.corr_remove_n))
    remove_n = min(remove_n, max(0, len(input_cols) - 1))  # keep at least one input node
    scores = compute_train_abs_pearson_scores(segments, input_cols, target_col, cfg)
    ranked = sorted(input_cols, key=lambda c: (scores.get(c, 0.0), c))
    removed = ranked[:remove_n]
    kept = [c for c in input_cols if c not in set(removed)]
    meta = {
        "enabled": bool(cfg.use_corr_filter or remove_n > 0),
        "remove_n": int(remove_n),
        "scores_abs_pearson_train": scores,
        "ranked_low_to_high": ranked,
        "removed_cols": removed,
        "kept_cols": kept,
    }
    return kept, meta

# =========================================================
# Target transform
# =========================================================

def resolve_target_transform(target_col: str, cfg: Config) -> str:
    if cfg.target_transform == "auto":
        # NO2/NOx 目标范围较稳定，默认不做 log；CO/C6H6 可选 log1p 防止长尾。
        return "log1p" if target_col in ("CO_GT", "C6H6_GT") else "none"
    if cfg.target_transform not in ("none", "log1p"):
        raise ValueError("target_transform must be auto, none or log1p")
    return cfg.target_transform


def transform_target_y(y: np.ndarray, transform: str) -> np.ndarray:
    y = y.astype(np.float32)
    if transform == "none":
        return y
    if transform == "log1p":
        # log1p 只改变训练空间，最终指标仍在原始空间计算。
        return np.log1p(np.maximum(y, 0.0)).astype(np.float32)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_transform_target_y(y: np.ndarray, transform: str) -> np.ndarray:
    y = y.astype(np.float32)
    if transform == "none":
        return y
    if transform == "log1p":
        return np.expm1(y).astype(np.float32)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_transform_target_y_torch(y: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "none":
        return y
    if transform == "log1p":
        return torch.expm1(y)
    raise ValueError(f"Unknown target transform: {transform}")


# =========================================================
# Window generation
# =========================================================

def split_indices_with_guard(T: int, cfg: Config) -> Dict[str, Tuple[int, int]]:
    total = cfg.split_train + cfg.split_val + cfg.split_test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    guard = cfg.window - 1
    raw_train_end = int(T * cfg.split_train)
    raw_val_end = int(T * (cfg.split_train + cfg.split_val))

    out = {
        "train": (0, raw_train_end),
        "val": (raw_train_end + guard, raw_val_end),
        "test": (raw_val_end + guard, T),
    }
    for k, (s, e) in out.items():
        if e - s < cfg.window:
            raise ValueError(f"{k} slice too short. T={T}, slice=({s},{e}), window={cfg.window}")
    return out


def generate_windows_from_slice(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    start: int,
    end: int,
    cfg: Config,
    split_name: str,
) -> Dict[str, np.ndarray]:
    sub = df.iloc[start:end].reset_index(drop=False)
    X_raw = sub[input_cols].to_numpy(dtype=np.float32)
    y_raw = sub[target_col].to_numpy(dtype=np.float32)
    seg_raw = sub["__segment_id__"].to_numpy(dtype=np.int64)
    time_raw = sub["__local_time__"].to_numpy(dtype=np.int64)

    X_list, y_list, seg_list, time_list = [], [], [], []
    skipped_nan_y = 0
    skipped_nan_x = 0
    for t in range(cfg.window - 1, len(sub), cfg.stride):
        y_t = y_raw[t]
        if not np.isfinite(y_t):
            skipped_nan_y += 1
            continue
        xw = X_raw[t - cfg.window + 1:t + 1, :].T  # [D,W]
        if not np.isfinite(xw).all():
            skipped_nan_x += 1
            continue
        X_list.append(xw.astype(np.float32))
        y_list.append(float(y_t))
        seg_list.append(int(seg_raw[t]))
        time_list.append(int(time_raw[t]))

    if not X_list:
        raise ValueError(
            f"No valid windows generated for {split_name}, slice=({start},{end}), window={cfg.window}, "
            f"skipped_nan_y={skipped_nan_y}, skipped_nan_x={skipped_nan_x}"
        )

    return {
        "X": np.stack(X_list, axis=0),
        "y": np.asarray(y_list, dtype=np.float32),
        "seg_id": np.asarray(seg_list, dtype=np.int64),
        "time_index": np.asarray(time_list, dtype=np.int64),
        "skipped_nan_y": np.asarray([skipped_nan_y], dtype=np.int64),
        "skipped_nan_x": np.asarray([skipped_nan_x], dtype=np.int64),
    }


def build_windows_segment_split(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Dict[str, Any]:
    buffers = {s: {"X": [], "y": [], "seg_id": [], "time_index": []} for s in ["train", "val", "test"]}
    split_meta: Dict[str, Any] = {"mode": "segment", "segments": []}

    for seg_id, df in enumerate(segments):
        idx = split_indices_with_guard(len(df), cfg)
        seg_meta = {"segment_id": seg_id, "num_rows": int(len(df)), "slices": {}}
        for split_name in ["train", "val", "test"]:
            s, e = idx[split_name]
            out = generate_windows_from_slice(df, input_cols, target_col, s, e, cfg, split_name)
            for key in buffers[split_name]:
                buffers[split_name][key].append(out[key])
            seg_meta["slices"][split_name] = [int(s), int(e)]
            seg_meta[f"{split_name}_windows"] = int(len(out["y"]))
        split_meta["segments"].append(seg_meta)

    merged = {}
    for split_name in ["train", "val", "test"]:
        merged[split_name] = {k: np.concatenate(v, axis=0) for k, v in buffers[split_name].items()}

    return {"splits": merged, "meta": split_meta}



def build_windows_segment_blocked_split(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Dict[str, Any]:
    """
    每个年度段前 split_train 作为训练集；后半段按固定长度 block 交替划为 val/test。
    这样 val/test 都覆盖后半段不同工况，避免原始 60/20/20 连续切分导致验证集与测试集分布差异过大。
    每个 block 内单独滑窗，不跨 val/test block 边界。
    """
    buffers = {s: {"X": [], "y": [], "seg_id": [], "time_index": []} for s in ["train", "val", "test"]}
    split_meta: Dict[str, Any] = {"mode": "segment_blocked", "block_size": int(cfg.block_size), "segments": []}
    guard = cfg.window - 1

    for seg_id, df in enumerate(segments):
        T = len(df)
        train_end = int(T * cfg.split_train)
        seg_meta = {"segment_id": seg_id, "num_rows": int(T), "train_slice": [0, int(train_end)], "val_slices": [], "test_slices": []}

        # train: 连续前段
        out = generate_windows_from_slice(df, input_cols, target_col, 0, train_end, cfg, "train")
        for key in buffers["train"]:
            buffers["train"][key].append(out[key])
        seg_meta["train_windows"] = int(len(out["y"]))

        # val/test: 后段交替 block。post_start 加 guard，避免训练结尾信息进入验证/测试窗口。
        post_start = train_end + guard
        block_size = max(int(cfg.block_size), cfg.window)
        block_id = 0
        for s0 in range(post_start, T, block_size):
            e0 = min(s0 + block_size, T)
            if e0 - s0 < cfg.window:
                continue
            split_name = "val" if (block_id % 2 == 0) else "test"
            out = generate_windows_from_slice(df, input_cols, target_col, s0, e0, cfg, split_name)
            for key in buffers[split_name]:
                buffers[split_name][key].append(out[key])
            seg_meta[f"{split_name}_slices"].append([int(s0), int(e0)])
            block_id += 1

        seg_meta["val_windows"] = int(sum(len(x) for x in buffers["val"]["y"] if len(x) > 0 and len(buffers["val"]["y"]) >= 0))
        seg_meta["test_windows"] = int(sum(len(x) for x in buffers["test"]["y"] if len(x) > 0 and len(buffers["test"]["y"]) >= 0))
        split_meta["segments"].append(seg_meta)

    merged = {}
    for split_name in ["train", "val", "test"]:
        if len(buffers[split_name]["y"]) == 0:
            raise ValueError(f"No {split_name} windows generated. Try smaller --window or --block_size")
        merged[split_name] = {k: np.concatenate(v, axis=0) for k, v in buffers[split_name].items()}
    return {"splits": merged, "meta": split_meta}


def build_windows_global_split(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Dict[str, Any]:
    df_all = pd.concat(segments, axis=0, ignore_index=True)
    df_all["__segment_id__"] = 0
    df_all["__local_time__"] = np.arange(len(df_all), dtype=np.int64)
    idx = split_indices_with_guard(len(df_all), cfg)
    merged = {}
    split_meta: Dict[str, Any] = {"mode": "global", "num_rows": int(len(df_all)), "slices": {}}
    for split_name in ["train", "val", "test"]:
        s, e = idx[split_name]
        out = generate_windows_from_slice(df_all, input_cols, target_col, s, e, cfg, split_name)
        merged[split_name] = out
        split_meta["slices"][split_name] = [int(s), int(e)]
        split_meta[f"{split_name}_windows"] = int(len(out["y"]))
    return {"splits": merged, "meta": split_meta}


class Standardizer:
    def __init__(self):
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.x_std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.x_std = np.where(self.x_std < 1e-8, 1.0, self.x_std)
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(np.std(y_train))
        if self.y_std < 1e-8:
            self.y_std = 1.0

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_std: np.ndarray) -> np.ndarray:
        return (y_std * self.y_std + self.y_mean).astype(np.float32)

    def meta(self) -> Dict[str, Any]:
        return {
            "x_mean": self.x_mean.reshape(-1).tolist(),
            "x_std": self.x_std.reshape(-1).tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seg_id: np.ndarray, time_index: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seg_id = torch.from_numpy(seg_id).long()
        self.time_index = torch.from_numpy(time_index).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.seg_id[idx], self.time_index[idx]


def preprocess_missing_for_target(
    segments: List[pd.DataFrame],
    input_cols: List[str],
    target_col: str,
    cfg: Config,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Missing-value protocol for UCI Air Quality.
    - load_airquality_file() has already converted -200 to NaN.
    - Inputs are online sensor/environment variables: interpolate by time by default.
    - Target labels are not interpolated by default; windows whose current y is NaN are skipped.
    """
    out_segments: List[pd.DataFrame] = []
    meta: Dict[str, Any] = {
        "fill_missing": cfg.fill_missing,
        "target_fill": cfg.target_fill,
        "input_cols": input_cols,
        "target_col": target_col,
        "before_missing_count": {},
        "after_missing_count": {},
    }

    for seg_id, df0 in enumerate(segments):
        df = df0.copy()
        cols_to_track = list(dict.fromkeys(input_cols + [target_col]))
        meta["before_missing_count"][str(seg_id)] = {c: int(df[c].isna().sum()) for c in cols_to_track}

        if cfg.fill_missing == "interpolate":
            # Only fill input/process variables. This uses past and future values inside the same continuous series.
            # It is acceptable for offline benchmark preprocessing and prevents fake extreme -200 values entering the model.
            df[input_cols] = (
                df[input_cols]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        elif cfg.fill_missing == "drop":
            # Do not fill inputs; windows containing NaN inputs will be skipped in generate_windows_from_slice().
            pass
        else:
            raise ValueError("fill_missing must be interpolate or drop")

        if cfg.target_fill == "interpolate":
            # Optional, not recommended for final strict experiments.
            df[target_col] = (
                df[target_col]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        elif cfg.target_fill == "drop":
            # Strict default: keep target NaN; only current-time missing labels are skipped.
            pass
        else:
            raise ValueError("target_fill must be drop or interpolate")

        meta["after_missing_count"][str(seg_id)] = {c: int(df[c].isna().sum()) for c in cols_to_track}
        out_segments.append(df)

    return out_segments, meta


def prepare_target_data(
    segments: List[pd.DataFrame],
    target_col: str,
    cfg: Config,
) -> Dict[str, Any]:
    input_cols = get_input_columns(cfg, target_col)
    for c in input_cols + [target_col]:
        if c not in REQUIRED_COLS:
            raise ValueError(f"Unknown column {c}. Valid={REQUIRED_COLS}")

    # Important: handle missing values before sliding-window generation.
    segments_used, missing_meta = preprocess_missing_for_target(segments, input_cols, target_col, cfg)

    corr_meta = {"enabled": False, "remove_n": 0, "scores_abs_pearson_train": {}, "ranked_low_to_high": [], "removed_cols": [], "kept_cols": input_cols}
    if cfg.use_corr_filter or int(cfg.corr_remove_n) > 0:
        input_cols, corr_meta = apply_correlation_screening(segments_used, input_cols, target_col, cfg)
        if len(input_cols) < 1:
            raise ValueError("Correlation screening removed all input columns; reduce --corr_remove_n")

    if cfg.split_mode == "segment":
        built = build_windows_segment_split(segments_used, input_cols, target_col, cfg)
    elif cfg.split_mode == "global":
        built = build_windows_global_split(segments_used, input_cols, target_col, cfg)
    elif cfg.split_mode == "segment_blocked":
        built = build_windows_segment_blocked_split(segments_used, input_cols, target_col, cfg)
    else:
        raise ValueError("split_mode must be segment, global or segment_blocked")

    splits = built["splits"]
    y_transform = resolve_target_transform(target_col, cfg)

    scaler = Standardizer()
    y_train_model = transform_target_y(splits["train"]["y"], y_transform)
    scaler.fit(splits["train"]["X"], y_train_model)

    prepared = {
        "target_col": target_col,
        "target_transform": y_transform,
        "input_cols": input_cols,
        "num_nodes": len(input_cols),
        "meta": {**built["meta"], "missing_preprocess": missing_meta, "correlation_screening": corr_meta},
        "scaler": scaler,
        "raw": {},
        "std": {},
    }
    for split in ["train", "val", "test"]:
        y_raw = splits[split]["y"]
        y_model = transform_target_y(y_raw, y_transform)
        prepared["raw"][f"X_{split}"] = splits[split]["X"]
        prepared["raw"][f"y_{split}"] = y_raw
        prepared["raw"][f"y_model_{split}"] = y_model
        prepared["raw"][f"seg_{split}"] = splits[split]["seg_id"]
        prepared["raw"][f"time_{split}"] = splits[split]["time_index"]
        prepared["std"][f"X_{split}"] = scaler.transform_X(splits[split]["X"])
        prepared["std"][f"y_{split}"] = scaler.transform_y(y_model)
    return prepared

def make_dataloaders(prepared: Dict[str, Any], cfg: Config, device: torch.device) -> Dict[str, DataLoader]:
    dsets = {}
    for split in ["train", "val", "test"]:
        dsets[split] = WindowDataset(
            prepared["std"][f"X_{split}"],
            prepared["std"][f"y_{split}"],
            prepared["raw"][f"seg_{split}"],
            prepared["raw"][f"time_{split}"],
        )
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    pin = device.type == "cuda"
    return {
        "train": DataLoader(dsets["train"], batch_size=cfg.batch_size, shuffle=True,
                            drop_last=False, num_workers=cfg.num_workers, pin_memory=pin, generator=gen),
        "val": DataLoader(dsets["val"], batch_size=cfg.batch_size, shuffle=False,
                          drop_last=False, num_workers=cfg.num_workers, pin_memory=pin),
        "test": DataLoader(dsets["test"], batch_size=cfg.batch_size, shuffle=False,
                           drop_last=False, num_workers=cfg.num_workers, pin_memory=pin),
    }


# =========================================================
# Metrics and Ridge diagnostic
# =========================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> Dict[str, float]:
    """只返回正式实验需要的四个指标。"""
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    yrange = max(float(np.max(y_true) - np.min(y_true)), 1e-12)
    nmae = mae / yrange * 100.0
    nrmse = rmse / yrange * 100.0

    # Air Quality 目标一般不会为 0；这里保留 eps 防止极端分母。
    denom = np.maximum(np.abs(y_true), cfg.mape_eps_denom)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"NMAE": float(nmae), "NRMSE": float(nrmse), "MAPE": float(mape), "R2": r2}

def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # X: [N,D,W] -> flattened with bias; y already standardized
    Xt = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    Xe = X_eval.reshape(X_eval.shape[0], -1).astype(np.float64)
    Xt = np.concatenate([Xt, np.ones((Xt.shape[0], 1), dtype=np.float64)], axis=1)
    Xe = np.concatenate([Xe, np.ones((Xe.shape[0], 1), dtype=np.float64)], axis=1)
    A = Xt.T @ Xt + alpha * np.eye(Xt.shape[1], dtype=np.float64)
    A[-1, -1] -= alpha  # bias 不正则
    b = Xt.T @ y_train.astype(np.float64)
    coef = np.linalg.solve(A, b)
    return (Xe @ coef).astype(np.float32)


def ridge_diagnostic(prepared: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    scaler: Standardizer = prepared["scaler"]
    pred_val_std = ridge_fit_predict(prepared["std"]["X_train"], prepared["std"]["y_train"], prepared["std"]["X_val"], alpha=1.0)
    pred_test_std = ridge_fit_predict(prepared["std"]["X_train"], prepared["std"]["y_train"], prepared["std"]["X_test"], alpha=1.0)
    pred_val = scaler.inverse_y(pred_val_std)
    pred_test = scaler.inverse_y(pred_test_std)
    return {
        "val": regression_metrics(prepared["raw"]["y_val"], pred_val, cfg),
        "test": regression_metrics(prepared["raw"]["y_test"], pred_test, cfg),
    }


# =========================================================
# Hypergraph helpers
# =========================================================

def build_hypergraph_adjacency_from_node_features_numpy(node_features: np.ndarray, k: int, eps: float = 1e-8) -> np.ndarray:
    if node_features.ndim != 2:
        raise ValueError(f"node_features must be [D,F], got {node_features.shape}")
    D, _ = node_features.shape
    if D < 2:
        raise ValueError("Need at least two nodes")
    k_eff = min(k, D - 1)
    diff = node_features[:, None, :] - node_features[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    delta = max(float(dist.mean()), eps)
    H = np.zeros((D, D), dtype=np.float32)
    edge_w = np.zeros((D,), dtype=np.float32)
    for j in range(D):
        idx = np.argsort(dist[:, j])[:k_eff + 1]
        H[idx, j] = 1.0
        incidence_w = np.exp(-(dist[idx, j] ** 2) / delta)
        edge_w[j] = float(np.mean(incidence_w))
    De = np.clip(H.sum(axis=0), 1.0, None)
    Dv = np.clip((H * edge_w[None, :]).sum(axis=1), eps, None)
    Dv_inv_sqrt = np.diag(np.power(Dv, -0.5))
    W_diag = np.diag(edge_w)
    De_inv = np.diag(np.power(De, -1.0))
    A = Dv_inv_sqrt @ H @ W_diag @ De_inv @ H.T @ Dv_inv_sqrt
    return A.astype(np.float32)


def build_static_hypergraph_from_train_windows(X_train_std: np.ndarray, k: int) -> np.ndarray:
    # X_train_std: [N,D,W]. 每个变量节点用其全部训练窗口轨迹展开作为节点特征。
    node_features = X_train_std.transpose(1, 0, 2).reshape(X_train_std.shape[1], -1)
    return build_hypergraph_adjacency_from_node_features_numpy(node_features, k=k)


# =========================================================
# Model
# =========================================================

class MultiViewMixerBlock(nn.Module):
    def __init__(self, num_nodes: int, window: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.ln_time = nn.LayerNorm(window)
        self.fc_time = nn.Linear(window, window)
        self.ln_node = nn.LayerNorm(num_nodes)
        self.fc_node1 = nn.Linear(num_nodes, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, num_nodes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,D,W]
        u = x + self.dropout(F.gelu(self.fc_time(self.ln_time(x))))
        z = u.transpose(1, 2)  # [B,W,D]
        z = z + self.dropout(self.fc_node2(F.gelu(self.fc_node1(self.ln_node(z)))))
        return z.transpose(1, 2)


class DynamicHypergraphBuilder(nn.Module):
    def __init__(self, k: int, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,D,F]
        B, D, _ = x.shape
        k_eff = min(self.k, D - 1)
        A_list = []
        for b in range(B):
            xb = x[b]
            dist = torch.cdist(xb, xb, p=2)
            delta = dist.mean().clamp(min=self.eps)
            knn_idx = torch.topk(dist, k=k_eff + 1, largest=False, dim=0).indices
            H = torch.zeros((D, D), device=x.device, dtype=x.dtype)
            edge_w = torch.zeros((D,), device=x.device, dtype=x.dtype)
            for j in range(D):
                members = knn_idx[:, j]
                H[members, j] = 1.0
                edge_w[j] = torch.exp(-(dist[members, j] ** 2) / delta).mean()
            De = H.sum(dim=0).clamp(min=1.0)
            Dv = (H * edge_w.unsqueeze(0)).sum(dim=1).clamp(min=self.eps)
            Dv_inv_sqrt = torch.diag(torch.pow(Dv, -0.5))
            W_diag = torch.diag(edge_w)
            De_inv = torch.diag(torch.pow(De, -1.0))
            A = Dv_inv_sqrt @ H @ W_diag @ De_inv @ H.T @ Dv_inv_sqrt
            A_list.append(A)
        return torch.stack(A_list, dim=0)


class InputEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.unsqueeze(1))  # [B,C,D,W]


class GatedTemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv_f = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, W = x.shape
        z = x.permute(0, 2, 1, 3).contiguous().view(B * D, C, W)
        z = F.pad(z, (self.left_pad, 0))
        out = self.conv_f(z) * torch.sigmoid(self.conv_g(z))
        out = self.dropout(out)
        return out.view(B, D, C, W).permute(0, 2, 1, 3).contiguous()



class MultiScaleGatedTemporalConv(nn.Module):
    """
    多尺度门控时序卷积。
    目的：同时捕捉 C6H6/NOx 等污染物浓度中的短时扰动与较长时间依赖。
    fusion='conv'  : 多尺度分支拼接后 1x1 卷积融合；
    fusion='gated' : 为每个尺度分支学习门控权重后加权融合。
    """
    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...],
        dilations: Tuple[int, ...],
        fusion: str,
        dropout: float,
    ):
        super().__init__()
        if fusion not in ("conv", "gated"):
            raise ValueError(f"multiscale_fusion must be conv or gated, got {fusion}")
        branches = []
        branch_desc = []
        for ks in kernel_sizes:
            for dl in dilations:
                ks = int(ks)
                dl = int(dl)
                if ks <= 0 or dl <= 0:
                    raise ValueError("kernel sizes and dilations must be positive")
                branches.append(GatedTemporalConv(channels, ks, dl, dropout))
                branch_desc.append((ks, dl))
        if not branches:
            raise ValueError("MultiScaleGatedTemporalConv needs at least one branch")
        self.branches = nn.ModuleList(branches)
        self.branch_desc = branch_desc
        self.num_branches = len(branches)
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout)
        if fusion == "conv":
            self.fuse = nn.Conv2d(channels * self.num_branches, channels, kernel_size=(1, 1), bias=True)
        else:
            self.gate = nn.Conv2d(channels * self.num_branches, self.num_branches, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        if self.fusion == "conv":
            return self.dropout(self.fuse(torch.cat(outs, dim=1)))
        cat = torch.cat(outs, dim=1)
        gate = torch.softmax(self.gate(cat), dim=1)  # [B, S, D, W]
        stacked = torch.stack(outs, dim=1)           # [B, S, C, D, W]
        return self.dropout((stacked * gate.unsqueeze(2)).sum(dim=1))



class FixedWaveletResidualEnhancer(nn.Module):
    """
    Fixed wavelet residual enhancement along temporal dimension.

    Input/Output: [B, D, W].
    The default fixed mode keeps the old behavior: a classical high-pass/detail filter is used
    to extract detail responses and inject them by residual connection:
        x_enhanced = x + alpha * detail

    The class also stores matched low-pass filters for the dual-filter learnable version.
    Supported wavelets: haar/db1, db2, db3, db4, sym2, sym3, coif1.
    """
    HI_FILTERS = {
        # decomposition high-pass filters, consistent with the previous implementation
        "haar": [-0.7071067811865476, 0.7071067811865476],
        "db1": [-0.7071067811865476, 0.7071067811865476],
        "db2": [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604],
        "db3": [-0.3326705529500826, 0.8068915093110928, -0.4598775021184915, -0.1350110200102546, 0.0854412738820267, 0.0352262918857095],
        "db4": [-0.2303778133088552, 0.7148465705529154, -0.6308807679298587, -0.0279837694168599, 0.1870348117188811, 0.0308413818355607, -0.0328830116668852, -0.0105974017850690],
        "sym2": [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604],
        "sym3": [-0.3326705529500826, 0.8068915093110928, -0.4598775021184915, -0.1350110200102546, 0.0854412738820267, 0.0352262918857095],
        "coif1": [-0.0727326195128539, 0.3378976624578092, -0.8525720202116004, 0.3848648468642029, 0.0727326195128539, -0.0156557281354645],
    }

    LO_FILTERS = {
        # decomposition low-pass filters. These are used to initialize the learnable low-frequency branch.
        "haar": [0.7071067811865476, 0.7071067811865476],
        "db1": [0.7071067811865476, 0.7071067811865476],
        "db2": [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341],
        "db3": [0.0352262918857095, -0.0854412738820267, -0.1350110200102546, 0.4598775021184915, 0.8068915093110928, 0.3326705529500826],
        "db4": [-0.0105974017850690, 0.0328830116668852, 0.0308413818355607, -0.1870348117188811, -0.0279837694168599, 0.6308807679298587, 0.7148465705529154, 0.2303778133088552],
        "sym2": [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341],
        "sym3": [0.0352262918857095, -0.0854412738820267, -0.1350110200102546, 0.4598775021184915, 0.8068915093110928, 0.3326705529500826],
        "coif1": [-0.0156557281354645, -0.0727326195128539, 0.3848648468642029, 0.8525720202116004, 0.3378976624578092, -0.0727326195128539],
    }

    # keep old attribute name for backward compatibility with existing code
    FILTERS = HI_FILTERS

    def __init__(self, alpha: float = 0.10, wavelet_type: str = "haar"):
        super().__init__()
        wt = str(wavelet_type).lower()
        if wt not in self.HI_FILTERS:
            raise ValueError(f"Unsupported wavelet_type={wavelet_type}. Supported: {sorted(self.HI_FILTERS)}")
        self.alpha = float(alpha)
        self.wavelet_type = wt
        hi = torch.tensor(self.HI_FILTERS[wt], dtype=torch.float32).view(1, 1, -1)
        lo = torch.tensor(self.LO_FILTERS[wt], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("detail_filter", hi)
        self.register_buffer("approx_filter", lo)

    def _normalize_filter(self, filt: torch.Tensor) -> torch.Tensor:
        return filt

    def get_detail_filter(self) -> torch.Tensor:
        return self._normalize_filter(self.detail_filter)

    def get_approx_filter(self) -> torch.Tensor:
        return self._normalize_filter(self.approx_filter)

    # backward-compatible alias
    def get_filter(self) -> torch.Tensor:
        return self.get_detail_filter()

    def get_alpha(self) -> torch.Tensor:
        return torch.tensor(float(self.alpha), dtype=self.detail_filter.dtype, device=self.detail_filter.device)

    @staticmethod
    def _same_length_conv(x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        # x: [B,D,W], filt: [1,1,L]
        B, D, W = x.shape
        L = int(filt.shape[-1])
        pad_left = (L - 1) // 2
        pad_right = (L - 1) - pad_left
        z = x.reshape(B * D, 1, W)
        z = F.pad(z, (pad_left, pad_right), mode="replicate")
        y = F.conv1d(z, filt).reshape(B, D, W)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,D,W]
        if self.alpha == 0.0:
            return x
        if x.shape[-1] < 2:
            return x
        detail = self._same_length_conv(x, self.get_detail_filter())
        return x + self.get_alpha() * detail


class LearnableWaveletResidualEnhancer(FixedWaveletResidualEnhancer):
    """
    QMF-constrained learnable wavelet residual enhancement.

    Important difference from the previous dual-independent-filter version:
      - only the low-pass prototype filter is an independent learnable Parameter;
      - the high-pass/detail filter is strictly generated from the low-pass prototype
        using the QMF (Quadrature Mirror Filter) relation:
            h_H[n] = (-1)^(n+1) h_L[K-1-n]
        The global sign convention does not affect the energy, but this convention
        matches the fixed high-pass filters stored in HI_FILTERS.

    Therefore, the high-pass filter is still learnable in an implicit/structural way:
    gradients from the prediction loss flow through the high-pass branch back to the
    low-pass prototype. This prevents the wavelet filters from degenerating into two
    unrelated ordinary convolution kernels.

    Forward residual:
        detail = Conv1D(x, QMF(h_L))
        approx = Conv1D(x, h_L)
        x_enhanced = x + alpha * (detail + lowpass_alpha_ratio * low_term)

    low_term:
        residual mode: approx - x
        direct mode:   approx

    Regularization:
      - energy: unit energy of h_L and QMF(h_L) -> 1.
    """
    def __init__(
        self,
        alpha: float = 0.10,
        wavelet_type: str = "haar",
        learnable_alpha: bool = False,
        alpha_max: float = 0.50,
        filter_norm: bool = True,
        learn_lowpass: bool = False,
        lowpass_mode: str = "residual",
        lowpass_alpha_ratio: float = 1.0,
    ):
        super().__init__(alpha=alpha, wavelet_type=wavelet_type)

        init_hi = self.detail_filter.detach().clone()
        init_lo = self.approx_filter.detach().clone()
        del self._buffers["detail_filter"]
        del self._buffers["approx_filter"]

        # QMF-constrained design: learn only the low-pass prototype.
        # The high-pass/detail filter is derived from this parameter in every forward pass.
        self.approx_filter = nn.Parameter(init_lo)
        self.register_buffer("initial_approx_filter", init_lo.clone())
        self.register_buffer("initial_detail_filter", init_hi.clone())

        # This flag now means whether the low-frequency approximation branch is injected.
        # Even when False, the low-pass prototype is still the learnable source from which
        # the high-pass filter is generated by QMF.
        self.learn_lowpass = bool(learn_lowpass)
        self.use_lowpass_branch = bool(learn_lowpass)

        self.learnable_alpha = bool(learnable_alpha)
        self.alpha_max = float(alpha_max)
        self.filter_norm = bool(filter_norm)
        self.lowpass_mode = str(lowpass_mode).lower()
        self.lowpass_alpha_ratio = float(lowpass_alpha_ratio)
        if self.lowpass_mode not in ("residual", "direct"):
            raise ValueError("wavelet_lowpass_mode must be residual or direct")

        # Sanity check: the QMF high-pass generated from the initial low-pass should
        # reproduce the classical high-pass filter up to numerical precision.
        with torch.no_grad():
            qmf_hi = self._qmf_from_lowpass(init_lo)
            self.initial_qmf_max_abs_error = float(torch.max(torch.abs(qmf_hi - init_hi)).cpu())

        if self.learnable_alpha:
            eps = 1e-6
            init_ratio = min(max(float(alpha) / max(self.alpha_max, eps), eps), 1.0 - eps)
            init_logit = np.log(init_ratio / (1.0 - init_ratio))
            self.alpha_logit = nn.Parameter(torch.tensor(float(init_logit), dtype=torch.float32))
        else:
            self.register_parameter("alpha_logit", None)

    @staticmethod
    def _qmf_from_lowpass(low_filter: torch.Tensor) -> torch.Tensor:
        """
        Generate the decomposition high-pass filter from the low-pass prototype.

        low_filter: [1, 1, K]
        high[n] = (-1)^(n+1) * low[K-1-n]

        With this sign convention, classical db2/sym2/coif1 low-pass filters generate
        the high-pass filters defined in HI_FILTERS.
        """
        high = torch.flip(low_filter, dims=[-1])
        signs = torch.ones_like(high)
        signs[..., 0::2] = -1.0
        signs[..., 1::2] = 1.0
        return high * signs

    def _normalize_filter(self, filt: torch.Tensor) -> torch.Tensor:
        if self.filter_norm:
            filt = filt / (filt.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8)
        return filt

    def get_approx_filter(self) -> torch.Tensor:
        return self._normalize_filter(self.approx_filter)

    def get_detail_filter(self) -> torch.Tensor:
        # Generate high-pass/detail filter from the normalized low-pass prototype.
        # This keeps high-pass and low-pass structurally tied during forward.
        lo = self.get_approx_filter()
        return self._qmf_from_lowpass(lo)

    # backward-compatible alias
    def get_filter(self) -> torch.Tensor:
        return self.get_detail_filter()

    def get_alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit) * float(self.alpha_max)
        return torch.tensor(float(self.alpha), dtype=self.approx_filter.dtype, device=self.approx_filter.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0 and not self.learnable_alpha:
            return x
        if x.shape[-1] < 2:
            return x

        hi = self.get_detail_filter()
        detail = self._same_length_conv(x, hi)
        residual = detail

        if self.use_lowpass_branch:
            lo = self.get_approx_filter()
            approx = self._same_length_conv(x, lo)
            if self.lowpass_mode == "residual":
                low_term = approx - x
            else:
                low_term = approx
            residual = residual + float(self.lowpass_alpha_ratio) * low_term

        return x + self.get_alpha() * residual

    def regularization_terms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return energy regularization loss."""
        lo_raw = self.approx_filter.view(-1)
        hi_raw = self._qmf_from_lowpass(self.approx_filter).view(-1)

        hi_norm = hi_raw / (torch.norm(hi_raw, p=2) + 1e-8)
        lo_norm = lo_raw / (torch.norm(lo_raw, p=2) + 1e-8)

        ortho_loss = torch.sum(hi_norm * lo_norm).pow(2)
        energy_loss = (torch.sum(hi_raw.pow(2)) - 1.0).pow(2) + (torch.sum(lo_raw.pow(2)) - 1.0).pow(2)
        return ortho_loss, energy_loss

    def regularization_loss(self, energy_lambda: float = 0.0) -> torch.Tensor:
        _, energy = self.regularization_terms()
        return float(energy_lambda) * energy

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        """Return learned low/high filters and structure diagnostics for summary.json."""
        lo_raw = self.approx_filter.detach().cpu().view(-1)
        hi_raw = self._qmf_from_lowpass(self.approx_filter.detach()).cpu().view(-1)
        lo_norm = lo_raw / (torch.norm(lo_raw, p=2) + 1e-8)
        hi_norm = hi_raw / (torch.norm(hi_raw, p=2) + 1e-8)
        qmf_hi = self._qmf_from_lowpass(self.approx_filter.detach()).cpu().view(-1)
        return {
            "wavelet_structure": "QMF-constrained learnable wavelet",
            "independent_parameter": "low-pass prototype hL",
            "derived_filter": "high-pass hH = QMF(hL)",
            "learned_low_filter": [float(v) for v in lo_raw.numpy().tolist()],
            "learned_high_filter": [float(v) for v in hi_raw.numpy().tolist()],
            "low_energy": float(torch.sum(lo_raw.pow(2)).item()),
            "high_energy": float(torch.sum(hi_raw.pow(2)).item()),
            "normalized_dot_low_high": float(torch.sum(lo_norm * hi_norm).item()),
            "energy_loss": float(((torch.sum(lo_raw.pow(2)) - 1.0).pow(2) + (torch.sum(hi_raw.pow(2)) - 1.0).pow(2)).item()),
            "initial_qmf_max_abs_error": float(getattr(self, "initial_qmf_max_abs_error", 0.0)),
            "current_qmf_max_abs_error": float(torch.max(torch.abs(qmf_hi - hi_raw)).item()),
            "use_lowpass_branch": bool(self.use_lowpass_branch),
            "lowpass_mode": str(self.lowpass_mode),
            "lowpass_alpha_ratio": float(self.lowpass_alpha_ratio),
            "filter_norm_in_forward": bool(self.filter_norm),
        }

    def extra_repr(self) -> str:
        mode = "learnable_alpha" if self.learnable_alpha else "fixed_alpha"
        branch = "with_lowpass_branch" if self.use_lowpass_branch else "detail_only_branch"
        return (
            f"wavelet_type={self.wavelet_type}, alpha={self.alpha}, {mode}, "
            f"QMF_constrained=True, independent_param=lowpass_prototype, "
            f"highpass=QMF(lowpass), filter_norm={self.filter_norm}, {branch}, "
            f"lowpass_mode={self.lowpass_mode}, lowpass_alpha_ratio={self.lowpass_alpha_ratio}"
        )

class HypergraphConv2D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.theta = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.theta(x)
        x_node = x.permute(0, 2, 1, 3).contiguous()  # [B,D,C,W]
        out = torch.einsum("bij,bjcw->bicw", A, x_node)
        out = out.permute(0, 2, 1, 3).contiguous()
        return self.dropout(F.gelu(out))


class TemporalHypergraphBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_multiscale_gtc: bool = False,
        multiscale_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        multiscale_dilations: Tuple[int, ...] = (1,),
        multiscale_fusion: str = "gated",
    ):
        super().__init__()
        if use_multiscale_gtc:
            self.tconv = MultiScaleGatedTemporalConv(
                channels=channels,
                kernel_sizes=multiscale_kernel_sizes,
                dilations=multiscale_dilations,
                fusion=multiscale_fusion,
                dropout=dropout,
            )
        else:
            self.tconv = GatedTemporalConv(channels, kernel_size, dilation, dropout)
        self.hconv = HypergraphConv2D(channels, dropout)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 残差式增强，避免图卷积破坏原始时序表征
        z = self.bn1(x + self.tconv(x))
        z = self.bn2(z + self.hconv(z, A))
        return z


class NodeAwareReadout(nn.Module):
    def __init__(self, channels: int, num_nodes: int, hidden: int, dropout: float):
        super().__init__()
        self.node_weight = nn.Parameter(torch.zeros(num_nodes))
        # features: weighted_mean(C) + global_mean(C) + global_max(C) + last_mean(C)
        self.fc1 = nn.Linear(channels * 4, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2 if hidden >= 2 else hidden)
        self.fc3 = nn.Linear(hidden // 2 if hidden >= 2 else hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,W]
        B, C, D, W = x.shape
        weights = torch.softmax(self.node_weight, dim=0)  # [D]
        weighted = (x * weights.view(1, 1, D, 1)).sum(dim=(2, 3))  # [B,C]
        gmean = x.mean(dim=(2, 3))
        gmax = x.amax(dim=(2, 3))
        last = x[..., -1].mean(dim=2)
        feat = torch.cat([weighted, gmean, gmax, last], dim=1)
        z = self.dropout(F.gelu(self.fc1(feat)))
        z = self.dropout(F.gelu(self.fc2(z)))
        return self.fc3(z).squeeze(-1)


class LWSTHGN(nn.Module):
    def __init__(self, num_nodes: int, window: int, cfg: Config, k: int, static_A: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        self.k = k
        self.num_nodes = num_nodes
        self.window = window
        self.mixers = nn.ModuleList([
            MultiViewMixerBlock(num_nodes, window, cfg.mixer_hidden, cfg.dropout)
            for _ in range(cfg.num_mixers)
        ])
        if cfg.use_wavelet_enhance:
            if bool(getattr(cfg, "wavelet_learnable", False)):
                self.wavelet_enhancer = LearnableWaveletResidualEnhancer(
                    alpha=cfg.wavelet_alpha,
                    wavelet_type=cfg.wavelet_type,
                    learnable_alpha=bool(getattr(cfg, "wavelet_learnable_alpha", False)),
                    alpha_max=float(getattr(cfg, "wavelet_alpha_max", 0.50)),
                    filter_norm=bool(getattr(cfg, "wavelet_filter_norm", True)),
                    learn_lowpass=bool(getattr(cfg, "wavelet_learnable_lowpass", False)),
                    lowpass_mode=str(getattr(cfg, "wavelet_lowpass_mode", "residual")),
                    lowpass_alpha_ratio=float(getattr(cfg, "wavelet_lowpass_alpha_ratio", 1.0)),
                )
            else:
                self.wavelet_enhancer = FixedWaveletResidualEnhancer(
                    alpha=cfg.wavelet_alpha,
                    wavelet_type=cfg.wavelet_type,
                )
        else:
            self.wavelet_enhancer = nn.Identity()
        self.embedding = InputEmbedding(cfg.embed_dim)
        self.blocks = nn.ModuleList([
            TemporalHypergraphBlock(
                channels=cfg.embed_dim,
                kernel_size=cfg.kernel_size,
                dilation=cfg.dilation,
                dropout=cfg.dropout,
                use_multiscale_gtc=cfg.use_multiscale_gtc,
                multiscale_kernel_sizes=cfg.multiscale_kernel_sizes,
                multiscale_dilations=cfg.multiscale_dilations,
                multiscale_fusion=cfg.multiscale_fusion,
            )
            for _ in range(cfg.num_blocks)
        ])
        self.readout = NodeAwareReadout(cfg.embed_dim, num_nodes, cfg.readout_hidden, cfg.dropout)

        if cfg.hypergraph_mode == "dynamic":
            self.graph_builder = DynamicHypergraphBuilder(k)
            self.register_buffer("static_A", torch.empty(0))
        elif cfg.hypergraph_mode == "static":
            if static_A is None:
                raise ValueError("static_A is required for static hypergraph")
            self.graph_builder = None
            self.register_buffer("static_A", torch.from_numpy(static_A).float())
        else:
            raise ValueError("hypergraph_mode must be static or dynamic")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,D,W]
        z = x
        if self.cfg.use_wavelet_enhance and self.cfg.wavelet_position == "pre_mixer":
            z = self.wavelet_enhancer(z)
        for mixer in self.mixers:
            z = mixer(z)
        if self.cfg.use_wavelet_enhance and self.cfg.wavelet_position == "post_mixer":
            z = self.wavelet_enhancer(z)

        if self.cfg.hypergraph_mode == "dynamic":
            # 使用 mixer 输出作为节点特征动态构图
            A = self.graph_builder(z)
        else:
            A = self.static_A.unsqueeze(0).expand(x.shape[0], -1, -1)

        h = self.embedding(z)
        for block in self.blocks:
            h = block(h, A)
        y = self.readout(h)
        return y, A


# =========================================================
# Train / eval
# =========================================================

def get_device(cfg: Config) -> torch.device:
    if cfg.gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.gpu}")
    return torch.device("cpu")


def compute_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config, prepared: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    if cfg.loss == "mse":
        return F.mse_loss(pred, target)
    if cfg.loss == "huber":
        return F.huber_loss(pred, target, delta=cfg.huber_delta)

    if cfg.loss in ("mixed_relative", "relative_huber"):
        if prepared is None:
            raise RuntimeError("prepared is required for relative loss")
        scaler: Standardizer = prepared["scaler"]
        transform = prepared.get("target_transform", "none")

        y_mean = torch.as_tensor(float(scaler.y_mean), device=pred.device, dtype=pred.dtype)
        y_std = torch.as_tensor(float(scaler.y_std), device=pred.device, dtype=pred.dtype)

        pred_model = pred * y_std + y_mean
        target_model = target * y_std + y_mean
        pred_raw = inverse_transform_target_y_torch(pred_model, transform)
        target_raw = inverse_transform_target_y_torch(target_model, transform)

        denom = torch.clamp(torch.abs(target_raw), min=float(cfg.relative_loss_denom))
        rel_err = (pred_raw - target_raw) / denom
        rel_loss = F.huber_loss(rel_err, torch.zeros_like(rel_err), delta=cfg.huber_delta)

        if cfg.loss == "relative_huber":
            return rel_loss

        base = F.huber_loss(pred, target, delta=cfg.huber_delta)
        return base + float(cfg.relative_loss_alpha) * rel_loss

    raise ValueError("loss must be mse, huber, mixed_relative or relative_huber")


def compute_wavelet_regularization(model: nn.Module, cfg: Config, base_loss: torch.Tensor) -> torch.Tensor:
    """Add energy regularization for learnable wavelet filters."""
    energy_lambda = float(getattr(cfg, "wavelet_energy_lambda", 0.0))
    if energy_lambda <= 0.0:
        return torch.zeros_like(base_loss)
    if not bool(getattr(cfg, "use_wavelet_enhance", False)):
        return torch.zeros_like(base_loss)
    enhancer = getattr(model, "wavelet_enhancer", None)
    if enhancer is None or not hasattr(enhancer, "regularization_loss"):
        return torch.zeros_like(base_loss)
    return enhancer.regularization_loss(energy_lambda=energy_lambda).to(base_loss.device)


@torch.no_grad()
def collect_wavelet_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """Collect learned QMF wavelet filters if the current model has a learnable wavelet enhancer."""
    enhancer = getattr(model, "wavelet_enhancer", None)
    if enhancer is None or not hasattr(enhancer, "diagnostics"):
        return {}
    try:
        return enhancer.diagnostics()
    except Exception as exc:
        return {"diagnostics_error": str(exc)}


def compute_validation_selection_score(
    val_mse: float,
    y_val_std: np.ndarray,
    pred_val_std: np.ndarray,
    prepared: Dict[str, Any],
    cfg: Config,
) -> Tuple[float, Dict[str, float]]:
    """
    std_mse: 原始逻辑，按标准化 MSE 选择；
    mape/nmae/nrmse: 越低越好；
    r2: 越高越好，内部取负号以便统一 min。
    """
    metric = str(cfg.selection_metric).lower()
    if metric == "std_mse":
        return float(val_mse), {}

    scaler: Standardizer = prepared["scaler"]
    transform = prepared.get("target_transform", "none")
    pred_model = scaler.inverse_y(pred_val_std)
    pred_raw = inverse_transform_target_y(pred_model, transform)
    y_raw = prepared["raw"]["y_val"]
    metrics = regression_metrics(y_raw, pred_raw, cfg)

    if metric == "mape":
        score = metrics["MAPE"]
    elif metric == "nmae":
        score = metrics["NMAE"]
    elif metric == "nrmse":
        score = metrics["NRMSE"]
    elif metric == "r2":
        score = -metrics["R2"]
    else:
        raise ValueError("selection_metric must be std_mse, mape, nmae, nrmse or r2")
    return float(score), metrics


@torch.no_grad()
def evaluate_std(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    preds, trues = [], []
    A_sum = None
    A_count = 0
    losses = []
    for X, y, _, _ in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred, A = model(X)
        loss = F.mse_loss(pred, y)
        losses.append(float(loss.detach().cpu()) * len(y))
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
        if A is not None:
            A_batch = A.detach().cpu().numpy()
            if A_sum is None:
                A_sum = A_batch.sum(axis=0)
            else:
                A_sum += A_batch.sum(axis=0)
            A_count += A_batch.shape[0]
    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    mse = float(np.sum(losses) / len(true_arr))
    A_mean = A_sum / max(A_count, 1) if A_sum is not None else None
    return mse, true_arr, pred_arr, A_mean


def train_one_setting(
    prepared: Dict[str, Any],
    cfg: Config,
    k: int,
    device: torch.device,
    target_out_dir: str,
) -> Dict[str, Any]:
    loaders = make_dataloaders(prepared, cfg, device)

    static_A = None
    if cfg.hypergraph_mode == "static":
        static_A = build_static_hypergraph_from_train_windows(prepared["std"]["X_train"], k=k)

    model = LWSTHGN(
        num_nodes=prepared["num_nodes"],
        window=cfg.window,
        cfg=cfg,
        k=k,
        static_A=static_A,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.max_epochs, 1), eta_min=cfg.lr * 0.05)

    best = {"epoch": -1, "val_mse": float("inf"), "score": float("inf"), "state": None, "val_metrics_at_select": {}}
    history = []
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for X, y, _, _ in loaders["train"]:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(X)
            base_loss = compute_loss(pred, y, cfg, prepared)
            reg_loss = compute_wavelet_regularization(model, cfg, base_loss)
            loss = base_loss + reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            total_n += len(y)
        scheduler.step()
        train_loss = total_loss / max(total_n, 1)
        val_mse, y_val_std_epoch, pred_val_std_epoch, _ = evaluate_std(model, loaders["val"], cfg, device)
        val_score, val_metrics_epoch = compute_validation_selection_score(
            val_mse, y_val_std_epoch, pred_val_std_epoch, prepared, cfg
        )
        hist_item = {"epoch": epoch, "train_loss": train_loss, "val_std_mse": val_mse, "selection_score": val_score, "lr": scheduler.get_last_lr()[0]}
        if val_metrics_epoch:
            hist_item.update({
                "val_NMAE": val_metrics_epoch["NMAE"],
                "val_NRMSE": val_metrics_epoch["NRMSE"],
                "val_MAPE": val_metrics_epoch["MAPE"],
                "val_R2": val_metrics_epoch["R2"],
            })
        history.append(hist_item)

        improved = val_score < best["score"] - cfg.early_stopping_min_delta
        if improved:
            best["epoch"] = epoch
            best["val_mse"] = val_mse
            best["score"] = val_score
            best["state"] = copy.deepcopy(model.state_dict())
            best["val_metrics_at_select"] = val_metrics_epoch
            no_improve = 0
            if cfg.verbose or epoch == 1:
                print(f"[{prepared['target_col']}] mode={cfg.hypergraph_mode} | k={k} | epoch {epoch:03d}/{cfg.max_epochs} | train_loss={train_loss:.6f} | val_std_mse={val_mse:.6f} | sel={cfg.selection_metric}:{val_score:.6f} | best_sel={best['score']:.6f} @ {best['epoch']}")
        else:
            no_improve += 1
            if cfg.verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"[{prepared['target_col']}] mode={cfg.hypergraph_mode} | k={k} | epoch {epoch:03d}/{cfg.max_epochs} | train_loss={train_loss:.6f} | val_std_mse={val_mse:.6f} | sel={cfg.selection_metric}:{val_score:.6f} | best_sel={best['score']:.6f} @ {best['epoch']}")

        if no_improve >= cfg.early_stopping_patience:
            if cfg.verbose:
                print(f"[{prepared['target_col']}] early stop at epoch {epoch}, best epoch {best['epoch']}")
            break

    model.load_state_dict(best["state"])
    val_mse, y_val_std, pred_val_std, A_val = evaluate_std(model, loaders["val"], cfg, device)
    test_mse, y_test_std, pred_test_std, A_test = evaluate_std(model, loaders["test"], cfg, device)

    scaler: Standardizer = prepared["scaler"]
    transform = prepared.get("target_transform", "none")
    # 模型输出先从标准化空间还原到训练目标空间，再还原到原始目标空间计算指标。
    pred_val_model = scaler.inverse_y(pred_val_std)
    pred_test_model = scaler.inverse_y(pred_test_std)
    pred_val = inverse_transform_target_y(pred_val_model, transform)
    pred_test = inverse_transform_target_y(pred_test_model, transform)
    y_val = prepared["raw"]["y_val"]
    y_test = prepared["raw"]["y_test"]

    val_metrics = regression_metrics(y_val, pred_val, cfg)
    test_metrics = regression_metrics(y_test, pred_test, cfg)

    wavelet_diagnostics = collect_wavelet_diagnostics(model)

    result = {
        "k": k,
        "best_epoch": int(best["epoch"]),
        "best_val_std_mse": float(best["val_mse"]),
        "best_selection_metric": str(cfg.selection_metric),
        "best_selection_score": float(best["score"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "model_state": best["state"] if cfg.save_model else None,
        "A_test_mean": A_test,
        "wavelet_diagnostics": wavelet_diagnostics,
        "predictions": {
            "val_true": y_val,
            "val_pred": pred_val,
            "test_true": y_test,
            "test_pred": pred_test,
        }
    }
    return result


def run_target(
    segments: List[pd.DataFrame],
    target_col: str,
    cfg: Config,
    device: torch.device,
    run_out_dir: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"Preparing target: {target_col} | window={cfg.window}")
    prepared = prepare_target_data(segments, target_col, cfg)
    corr_suffix = f"_corrRm{int(cfg.corr_remove_n)}" if (cfg.use_corr_filter or int(cfg.corr_remove_n) > 0) else ""
    target_out_dir = os.path.join(run_out_dir, f"{target_col}_W{cfg.window}{corr_suffix}")
    ensure_dir(target_out_dir)

    print(f"Input columns ({len(prepared['input_cols'])}): {prepared['input_cols']}")
    if cfg.use_multiscale_gtc:
        print(f"Multi-scale GTC: kernels={list(cfg.multiscale_kernel_sizes)} | dilations={list(cfg.multiscale_dilations)} | fusion={cfg.multiscale_fusion}")
    if cfg.use_wavelet_enhance:
        wav_mode = "learnable" if bool(getattr(cfg, "wavelet_learnable", False)) else "fixed"
        alpha_mode = "learnable" if bool(getattr(cfg, "wavelet_learnable_alpha", False)) else "fixed"
        print(f"Wavelet residual enhancement: mode={wav_mode} | type={cfg.wavelet_type} | alpha={cfg.wavelet_alpha} ({alpha_mode}) | position={cfg.wavelet_position}")
        if bool(getattr(cfg, "wavelet_learnable", False)):
            print(f"QMF learnable DWT filters: low-pass prototype=learnable | high-pass=QMF(low-pass) | low_freq_branch={bool(getattr(cfg, 'wavelet_learnable_lowpass', False))} | low_mode={getattr(cfg, 'wavelet_lowpass_mode', 'residual')} | low_ratio={getattr(cfg, 'wavelet_lowpass_alpha_ratio', 1.0)}")
            print(f"Wavelet constraints: energy_lambda={getattr(cfg, 'wavelet_energy_lambda', 0.0)}")
    corr_meta = prepared.get("meta", {}).get("correlation_screening", {})
    if corr_meta.get("enabled", False):
        print(f"Correlation screening: remove_n={corr_meta.get('remove_n')} | removed={corr_meta.get('removed_cols')}")
        score_items = sorted(corr_meta.get("scores_abs_pearson_train", {}).items(), key=lambda kv: kv[1])
        print("Train abs Pearson scores low->high: " + ", ".join([f"{k}:{v:.4f}" for k, v in score_items]))
    print(f"Target transform: {prepared['target_transform']}")
    print(f"Window: {cfg.window}, split_mode: {cfg.split_mode}, hypergraph_mode: {cfg.hypergraph_mode}")
    print(f"Num windows: train={len(prepared['std']['y_train'])}, val={len(prepared['std']['y_val'])}, test={len(prepared['std']['y_test'])}")

    if cfg.print_baseline:
        ridge = ridge_diagnostic(prepared, cfg)
        print("Ridge baseline only for diagnosis, not used by LWST-HGN:")
        print(json.dumps({"VAL_R2": ridge["val"]["R2"], "TEST_R2": ridge["test"]["R2"]}, ensure_ascii=False, indent=2))

    k_list = list(cfg.k_candidates) if cfg.auto_k_search else [cfg.fixed_k]
    results = []
    for k in k_list:
        set_global_seed(cfg.seed)
        res = train_one_setting(prepared, cfg, k, device, target_out_dir)
        results.append(res)

    best_res = min(results, key=lambda r: r.get("best_selection_score", r["best_val_std_mse"]))

    meta = {
        "target_col": target_col,
        "target_transform": prepared["target_transform"],
        "input_cols": prepared["input_cols"],
        "num_nodes": prepared["num_nodes"],
        "config": asdict(cfg),
        "data_meta": prepared["meta"],
        "scaler": prepared["scaler"].meta(),
        "all_k_summary": [
            {
                "k": r["k"],
                "best_epoch": r["best_epoch"],
                "best_val_std_mse": r["best_val_std_mse"],
                "best_selection_metric": r.get("best_selection_metric", str(cfg.selection_metric)),
                "best_selection_score": r.get("best_selection_score", r["best_val_std_mse"]),
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
            } for r in results
        ],
        "selected_k": best_res["k"],
        "best_epoch": best_res["best_epoch"],
        "best_selection_metric": best_res.get("best_selection_metric", str(cfg.selection_metric)),
        "best_selection_score": best_res.get("best_selection_score", best_res["best_val_std_mse"]),
        "best_selection_metric": best_res.get("best_selection_metric", str(cfg.selection_metric)),
        "best_selection_score": best_res.get("best_selection_score", best_res["best_val_std_mse"]),
        "correlation_screening": prepared.get("meta", {}).get("correlation_screening", {}),
        "val_metrics": best_res["val_metrics"],
        "test_metrics": best_res["test_metrics"],
        "wavelet_diagnostics": best_res.get("wavelet_diagnostics", {}),
    }

    save_json(meta, os.path.join(target_out_dir, "summary.json"))
    save_json(best_res["history"], os.path.join(target_out_dir, "history_best_k.json"))
    if cfg.save_predictions:
        pred = best_res["predictions"]
        np.savez(os.path.join(target_out_dir, "predictions.npz"), **pred)
        pd.DataFrame({
            "split": ["val"] * len(pred["val_true"]) + ["test"] * len(pred["test_true"]),
            "y_true": np.concatenate([pred["val_true"], pred["test_true"]]),
            "y_pred": np.concatenate([pred["val_pred"], pred["test_pred"]]),
        }).to_csv(os.path.join(target_out_dir, "predictions.csv"), index=False)
    if cfg.save_adjacency and best_res["A_test_mean"] is not None:
        np.save(os.path.join(target_out_dir, "test_mean_adjacency.npy"), best_res["A_test_mean"])
    if cfg.save_model and best_res["model_state"] is not None:
        torch.save(best_res["model_state"], os.path.join(target_out_dir, "best_model.pt"))

    print("\n" + "-" * 80)
    print(f"=== TARGET {target_col} SUMMARY ===")
    print(f"=== MODE: {cfg.hypergraph_mode} ===")
    print(f"=== WINDOW: {cfg.window} ===")
    print(f"=== SELECTED k: {best_res['k']} ===")
    print(f"=== SELECTION: {best_res.get('best_selection_metric', str(cfg.selection_metric))} | score={best_res.get('best_selection_score', best_res['best_val_std_mse']):.6f} ===")
    print("=== VAL METRICS ===")
    print(json.dumps(best_res["val_metrics"], ensure_ascii=False, indent=2))
    print("=== TEST METRICS ===")
    print(json.dumps(best_res["test_metrics"], ensure_ascii=False, indent=2))
    print("-" * 80)

    return {
        "target": target_col,
        "window": cfg.window,
        "mode": cfg.hypergraph_mode,
        "selected_k": best_res["k"],
        "best_epoch": best_res["best_epoch"],
        "best_selection_metric": best_res.get("best_selection_metric", str(cfg.selection_metric)),
        "best_selection_score": best_res.get("best_selection_score", best_res["best_val_std_mse"]),
        "corr_remove_n": int(prepared.get("meta", {}).get("correlation_screening", {}).get("remove_n", 0)),
        "removed_cols": prepared.get("meta", {}).get("correlation_screening", {}).get("removed_cols", []),
        "kept_cols": prepared.get("input_cols", []),
        "use_multiscale_gtc": bool(cfg.use_multiscale_gtc),
        "multiscale_kernel_sizes": list(cfg.multiscale_kernel_sizes) if cfg.use_multiscale_gtc else [],
        "multiscale_dilations": list(cfg.multiscale_dilations) if cfg.use_multiscale_gtc else [],
        "multiscale_fusion": cfg.multiscale_fusion if cfg.use_multiscale_gtc else "",
        "use_wavelet_enhance": bool(cfg.use_wavelet_enhance),
        "wavelet_type": cfg.wavelet_type if cfg.use_wavelet_enhance else "",
        "wavelet_alpha": float(cfg.wavelet_alpha) if cfg.use_wavelet_enhance else 0.0,
        "wavelet_position": cfg.wavelet_position if cfg.use_wavelet_enhance else "",
        "wavelet_learnable": bool(getattr(cfg, "wavelet_learnable", False)) if cfg.use_wavelet_enhance else False,
        "wavelet_learnable_alpha": bool(getattr(cfg, "wavelet_learnable_alpha", False)) if cfg.use_wavelet_enhance else False,
        "wavelet_alpha_max": float(getattr(cfg, "wavelet_alpha_max", 0.50)) if cfg.use_wavelet_enhance else 0.0,
        "wavelet_filter_norm": bool(getattr(cfg, "wavelet_filter_norm", True)) if cfg.use_wavelet_enhance else False,
        "wavelet_learnable_lowpass": bool(getattr(cfg, "wavelet_learnable_lowpass", False)) if cfg.use_wavelet_enhance else False,
        "wavelet_lowpass_mode": str(getattr(cfg, "wavelet_lowpass_mode", "residual")) if cfg.use_wavelet_enhance else "",
        "wavelet_lowpass_alpha_ratio": float(getattr(cfg, "wavelet_lowpass_alpha_ratio", 1.0)) if cfg.use_wavelet_enhance else 0.0,
        "wavelet_energy_lambda": float(getattr(cfg, "wavelet_energy_lambda", 0.0)) if cfg.use_wavelet_enhance else 0.0,
        "VAL_NMAE": best_res["val_metrics"]["NMAE"],
        "VAL_NRMSE": best_res["val_metrics"]["NRMSE"],
        "VAL_MAPE": best_res["val_metrics"]["MAPE"],
        "VAL_R2": best_res["val_metrics"]["R2"],
        "TEST_NMAE": best_res["test_metrics"]["NMAE"],
        "TEST_NRMSE": best_res["test_metrics"]["NRMSE"],
        "TEST_MAPE": best_res["test_metrics"]["MAPE"],
        "TEST_R2": best_res["test_metrics"]["R2"],
    }


# =========================================================
# CLI
# =========================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="./data/AirQualityUCI.csv")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--target", type=str, default="C6H6_GT", choices=["NO2_GT", "NOx_GT", "CO_GT", "C6H6_GT", "both"])
    p.add_argument("--input_cols", nargs="*", default=None)
    p.add_argument("--input_mode", type=str, default="strict", choices=["strict", "wide"])
    p.add_argument("--use_corr_filter", type=int, default=1, help="Enable training-set Pearson correlation screening")
    p.add_argument("--corr_remove_n", type=int, default=3, help="Remove n inputs with the lowest training-set absolute Pearson correlation")
    p.add_argument("--corr_remove_list", type=str, default="3", help="Comma-separated removal counts; released best uses 3")

    p.add_argument("--split_mode", type=str, default="segment", choices=["segment", "global", "segment_blocked"])
    p.add_argument("--split_train", type=float, default=0.60)
    p.add_argument("--split_val", type=float, default=0.20)
    p.add_argument("--split_test", type=float, default=0.20)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--window_list", type=str, default="3", help="Comma-separated windows; released best uses 3")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--block_size", type=int, default=168)
    p.add_argument("--fill_missing", type=str, default="interpolate", choices=["interpolate", "drop"])
    p.add_argument("--target_fill", type=str, default="drop", choices=["drop", "interpolate"])
    p.add_argument("--target_transform", type=str, default="log1p", choices=["auto", "none", "log1p"])

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--early_stopping_patience", type=int, default=25)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-7)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--loss", type=str, default="huber", choices=["mse", "huber", "mixed_relative", "relative_huber"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--relative_loss_alpha", type=float, default=0.3)
    p.add_argument("--relative_loss_denom", type=float, default=20.0)
    p.add_argument("--selection_metric", type=str, default="nrmse", choices=["std_mse", "mape", "nmae", "nrmse", "r2"])

    p.add_argument("--num_mixers", type=int, default=2)
    p.add_argument("--mixer_hidden", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--readout_hidden", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--dilation", type=int, default=1)
    p.add_argument("--use_multiscale_gtc", type=int, default=1)
    p.add_argument("--multiscale_kernel_sizes", type=str, default="3,5,7")
    p.add_argument("--multiscale_dilations", type=str, default="1")
    p.add_argument("--multiscale_fusion", type=str, default="gated", choices=["conv", "gated"])
    p.add_argument("--use_wavelet_enhance", type=int, default=1, help="Enable learnable wavelet residual enhancement")
    p.add_argument("--wavelet_type", type=str, default="db2", choices=["haar", "db1", "db2", "db3", "db4", "sym2", "sym3", "coif1"], help="Classical wavelet used to initialize the learnable low-pass prototype")
    p.add_argument("--wavelet_alpha", type=float, default=0.055, help="Wavelet residual enhancement scale alpha")
    p.add_argument("--wavelet_position", type=str, default="post_mixer", choices=["pre_mixer", "post_mixer"], help="小波增强位置")
    p.add_argument("--wavelet_learnable", type=int, default=1, help="1: learnable low-pass prototype with QMF-derived high-pass filter")
    p.add_argument("--wavelet_learnable_alpha", type=int, default=0, help="1: alpha 也可学习；0: alpha 固定为 wavelet_alpha")
    p.add_argument("--wavelet_alpha_max", type=float, default=0.50, help="learnable alpha 的上界")
    p.add_argument("--wavelet_filter_norm", type=int, default=1, help="1: 每次 forward 对可学习滤波器做能量归一化")
    p.add_argument("--wavelet_learnable_lowpass", type=int, default=1, help="Enable low-frequency residual branch")
    p.add_argument("--wavelet_lowpass_mode", type=str, default="residual", choices=["residual", "direct"], help="低频分支融合方式：residual 表示 cA-x，direct 表示 cA")
    p.add_argument("--wavelet_lowpass_alpha_ratio", type=float, default=0.5, help="Low-frequency residual ratio rho")
    p.add_argument("--wavelet_energy_lambda", type=float, default=1e-4, help="Unit-energy regularization weight")

    p.add_argument("--hypergraph_mode", type=str, default="static", choices=["static", "dynamic"])
    p.add_argument("--auto_k_search", type=int, default=1)
    p.add_argument("--k_candidates", type=str, default="3")
    p.add_argument("--fixed_k", type=int, default=3)

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_predictions", type=int, default=1)
    p.add_argument("--save_adjacency", type=int, default=0)
    p.add_argument("--save_model", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--print_baseline", type=int, default=0)
    p.add_argument("--torch_num_threads", type=int, default=4)

    p.add_argument("--mape_zero_threshold", type=float, default=1e-6)
    p.add_argument("--mape_eps_denom", type=float, default=1e-6)

    a = p.parse_args()
    cfg = Config()
    for k, v in vars(a).items():
        setattr(cfg, k, v)
    cfg.auto_k_search = bool(cfg.auto_k_search)
    cfg.use_corr_filter = bool(cfg.use_corr_filter)
    cfg.save_predictions = bool(cfg.save_predictions)
    cfg.save_adjacency = bool(cfg.save_adjacency)
    cfg.save_model = bool(cfg.save_model)
    cfg.verbose = bool(cfg.verbose)
    cfg.print_baseline = bool(cfg.print_baseline)
    cfg.k_candidates = tuple(int(x.strip()) for x in str(cfg.k_candidates).split(",") if x.strip())
    cfg.use_multiscale_gtc = bool(cfg.use_multiscale_gtc)
    cfg.multiscale_kernel_sizes = tuple(int(x.strip()) for x in str(cfg.multiscale_kernel_sizes).split(",") if x.strip())
    cfg.multiscale_dilations = tuple(int(x.strip()) for x in str(cfg.multiscale_dilations).split(",") if x.strip())
    cfg.use_wavelet_enhance = bool(cfg.use_wavelet_enhance)
    cfg.wavelet_type = str(cfg.wavelet_type).lower()
    cfg.wavelet_learnable = bool(cfg.wavelet_learnable)
    cfg.wavelet_learnable_alpha = bool(cfg.wavelet_learnable_alpha)
    cfg.wavelet_filter_norm = bool(cfg.wavelet_filter_norm)
    cfg.wavelet_learnable_lowpass = bool(getattr(cfg, "wavelet_learnable_lowpass", False))
    cfg.wavelet_lowpass_mode = str(getattr(cfg, "wavelet_lowpass_mode", "residual")).lower()
    cfg.wavelet_lowpass_alpha_ratio = float(getattr(cfg, "wavelet_lowpass_alpha_ratio", 1.0))
    cfg.wavelet_energy_lambda = float(getattr(cfg, "wavelet_energy_lambda", 0.0))
    return cfg


def parse_window_list(cfg: Config) -> List[int]:
    if cfg.window_list and str(cfg.window_list).strip():
        wins = [int(x.strip()) for x in str(cfg.window_list).split(",") if x.strip()]
    else:
        wins = [int(cfg.window)]
    out = []
    for w in wins:
        if w <= 0:
            raise ValueError("window must be positive")
        if w not in out:
            out.append(w)
    return out


def parse_corr_remove_list(cfg: Config) -> List[int]:
    if cfg.corr_remove_list and str(cfg.corr_remove_list).strip():
        vals = [int(x.strip()) for x in str(cfg.corr_remove_list).split(",") if x.strip()]
    else:
        vals = [int(cfg.corr_remove_n)]
    out: List[int] = []
    for v in vals:
        v = max(0, int(v))
        if v not in out:
            out.append(v)
    return out


def main():
    cfg = parse_args()
    if int(cfg.torch_num_threads) > 0:
        torch.set_num_threads(int(cfg.torch_num_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(cfg.torch_num_threads))))
        except RuntimeError:
            pass
    set_global_seed(cfg.seed)
    device = get_device(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_tag = cfg.window_list if cfg.window_list else str(cfg.window)
    window_tag = str(window_tag).replace(",", "-")
    data_tag = os.path.splitext(os.path.basename(cfg.data_path))[0]
    corr_tag = "corr" if (cfg.use_corr_filter or cfg.corr_remove_list or int(cfg.corr_remove_n) > 0) else "nocorr"
    ms_tag = "MS" if cfg.use_multiscale_gtc else "SS"
    if cfg.use_wavelet_enhance and bool(getattr(cfg, "wavelet_learnable", False)):
        wav_tag = "LWAV"
    else:
        wav_tag = "WAV" if cfg.use_wavelet_enhance else "NOWAV"
    run_name = f"lw_sthgn_airquality_{data_tag}_{cfg.target}_{cfg.hypergraph_mode}_{ms_tag}_{wav_tag}_W{window_tag}_{cfg.split_mode}_{corr_tag}_{timestamp}"
    run_out_dir = os.path.join(cfg.out_dir, run_name)
    ensure_dir(run_out_dir)

    print("=" * 80)
    print("LWST-HGN UCI Air Quality / C6H6_GT")
    print("Direct soft-sensing prediction without a residual baseline.")
    print("Final metrics: NMAE, NRMSE, MAPE, R2")
    print(f"Device: {device}")
    print(f"Output dir: {run_out_dir}")
    print("=" * 80)

    segments, data_meta = load_airquality_file(cfg)
    save_json({"config": asdict(cfg), "data_meta": data_meta}, os.path.join(run_out_dir, "run_config_and_data_meta.json"))

    print("Loaded Air Quality file:")
    print(f"  file={data_meta['file']} | rows={data_meta['num_rows']} | time={data_meta['datetime_start']} -> {data_meta['datetime_end']}")
    print("Missing rates after replacing -200 with NaN:")
    for c, r in data_meta["raw_missing_rate"].items():
        print(f"  {c:14s}: {r*100:5.1f}%")

    targets = TARGET_COLS if cfg.target == "both" else [cfg.target]
    windows = parse_window_list(cfg)
    corr_remove_values = parse_corr_remove_list(cfg)
    compact_all = []

    for target_col in targets:
        target_results = []
        for w in windows:
            for rm in corr_remove_values:
                cfg_w = copy.deepcopy(cfg)
                cfg_w.window = int(w)
                cfg_w.corr_remove_n = int(rm)
                cfg_w.use_corr_filter = bool(cfg.use_corr_filter or int(rm) > 0 or len(corr_remove_values) > 1)
                compact = run_target(segments, target_col, cfg_w, device, run_out_dir)
                target_results.append(compact)
        # If multiple settings are provided, retain the one with the highest validation R2.
        best_target = max(target_results, key=lambda x: x["VAL_R2"])
        compact_all.append(best_target)

    print("\n" + "=" * 80)
    print("ALL TARGETS COMPACT RESULTS")
    print("=" * 80)
    if len(compact_all) == 1:
        print(json.dumps(compact_all[0], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compact_all, ensure_ascii=False, indent=2))
    save_json(compact_all, os.path.join(run_out_dir, "compact_metrics.json"))


if __name__ == "__main__":
    main()
