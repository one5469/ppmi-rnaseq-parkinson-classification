#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPMI RNA-seq TabNet self-supervised Encoder pretraining-only script.

이 스크립트는 기존 PPMI RNA-seq TPM + metadata 병합 방식을 유지하면서,
TabNet Encoder를 masked feature reconstruction 방식으로 자기 지도 사전학습합니다.

변경 사항
---------
1. TabNetClassifier fine-tuning 및 scratch baseline 학습을 제거했습니다.
2. fold별 TabNetPretrainer 학습 후 pretrained Encoder 가중치를 별도로 저장합니다.
3. downstream CV에서 사용할 수 있도록 validation subject를 pretraining에서도 제외하는 구조는 유지합니다.
4. metadata DataFrame에서 gene feature column을 제거하고, feature matrix는 float32 기반으로 생성해 RAM 사용량을 줄였습니다.
5. fold 종료 시 큰 객체와 CUDA cache를 정리하도록 개선했습니다.
6. latent extraction은 선택 사항이며 기본값은 비활성화했습니다.

사용 예시
---------
CUDA_VISIBLE_DEVICES=3 python ppmi_tabnet_ssl_pretrain_only_optimized.py \
  --tpm-path ./Dataset/RNA/IR3/quant/tpm_gene_matrix.csv \
  --meta-path ./Dataset/RNA/IR3/metaDataIR3.csv \
  --out-dir ./results_tabnet_ssl_pretrain_only \
  --pretrain-groups PD Control Prodromal SWEDD \
  --target-groups PD Control \
  --n-splits 5 \
  --feature-filter variance \
  --top-n-features 5000 \
  --pretraining-ratio 0.3

저장 산출물
----------
각 fold 디렉토리에는 다음 파일이 저장됩니다.
- tabnet_pretrainer_full.zip: pytorch-tabnet Pretrainer 전체 모델
- pretrained_encoder_bundle.pt: encoder/embedder/network state_dict + feature/scaler 정보 bundle
- encoder_state_dict.pt: encoder state_dict만 별도 저장
- embedder_state_dict.pt: embedder가 존재할 경우 별도 저장
- network_state_dict.pt: --save-network-state-dict 사용 시 pretrainer network 전체 state_dict
- selected_feature_columns.csv: fold별 선택 feature 목록
- scaler.joblib: fold 내부 scaling을 사용한 경우 StandardScaler
- pretrainer_history.csv: 사전학습 history

주의
----
- 각 fold의 target validation subject는 pretraining에도 포함하지 않습니다.
- self-supervised pretraining에는 label을 사용하지 않습니다.
- 저장된 Encoder를 downstream classifier에 사용할 때는 동일한 selected_feature_columns 순서와 scaler를 적용해야 합니다.
"""

import argparse
import gc
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_to_jsonable(obj):
    """argparse Namespace 등을 JSON 저장 가능한 형태로 변환합니다."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): safe_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_to_jsonable(v) for v in obj]
    return obj


def save_history(model, path: Path) -> None:
    """pytorch-tabnet History 객체를 최대한 안전하게 CSV/JSON으로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hist = getattr(model, "history", None)
    if hist is None:
        return

    candidates = []
    if hasattr(hist, "history"):
        candidates.append(hist.history)
    candidates.append(hist)

    for item in candidates:
        try:
            df = pd.DataFrame(item)
            if len(df) > 0:
                df.to_csv(path, index=False)
                return
        except Exception:
            pass

    try:
        save_json(str(hist), path.with_suffix(".json"))
    except Exception:
        pass


def clear_cuda() -> None:
    """Python 객체와 PyTorch CUDA cache를 정리합니다."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_contiguous_float32(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """TabNet 입력에 적합하도록 contiguous float32 배열로 변환합니다."""
    if x is None:
        return None
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------------
# PPMI RNA-seq metadata normalization functions
# These functions follow the uploaded notebook style.
# ---------------------------------------------------------------------

def normalize_diagnosis_for_tabnet(x):
    """PPMI metadata의 진단명을 실험용 표준 그룹명으로 변환합니다."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    mapping = {
        "PD": "PD",
        "Control": "Control",
        "HC": "Control",
        "Healthy control": "Control",
        "Healthy Control": "Control",
        "Prodromal": "Prodromal",
        "SWEDD": "SWEDD",
        "Genetic Cohort": "Genetic Cohort",
        "Genetic Registry": "Genetic Registry",
        "PDPOOL": "PDPOOL",
        "HCPOOL": "Control",
    }
    return mapping.get(x, x)


def normalize_visit_for_filter(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def normalize_sample_id(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if re.fullmatch(r"\d+(\.0)?", x):
        x = x.replace(".0", "")
        x = x.lstrip("0")
        return x if x != "" else "0"

    return x


def normalize_patno(x):
    if pd.isna(x):
        return np.nan

    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def normalize_visit(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def extract_hudalphaid_or_nan(x):
    """metadata 또는 파일명 문자열에서 5104-SL-... 형식의 HudAlphaID를 추출합니다."""
    if pd.isna(x):
        return np.nan

    m = re.search(r"5104-SL-\d+", str(x))
    if m is None:
        return np.nan
    return m.group(0)


def ensure_hudalphaid_column(meta_df: pd.DataFrame, target_col: str = "HudAlphaID") -> pd.DataFrame:
    """
    metadata에 featureCounts raw count matrix와 매칭 가능한 HudAlphaID 컬럼을 보장합니다.
    이 스크립트에서는 DESeq를 직접 실행하지 않지만, downstream 추적성을 위해 유지합니다.
    """
    out = meta_df.copy()
    candidate_sources = []

    if target_col in out.columns:
        candidate_sources.append((target_col, out[target_col]))

    candidate_sources.append(("index", pd.Series(out.index, index=out.index)))

    preferred_cols = [
        "Specimen Bar Code",
        "SAMPLE_ID",
        "Sample ID",
        "sample_id",
        "HudAlphaID_meta",
        "File Name",
        "filename",
    ]

    for c in preferred_cols:
        if c in out.columns and c != target_col:
            candidate_sources.append((c, out[c]))

    for c in out.columns:
        low = str(c).lower()
        if any(k in low for k in ["hud", "alpha", "sample", "specimen", "barcode", "bar code"]):
            if c != target_col and c not in preferred_cols:
                candidate_sources.append((c, out[c]))

    best_name = None
    best_values = None
    best_n = -1

    for name, values in candidate_sources:
        try:
            extracted = values.apply(extract_hudalphaid_or_nan)
            n = int(extracted.notna().sum())
        except Exception:
            continue

        if n > best_n:
            best_name = name
            best_values = extracted
            best_n = n

    if best_n > 0:
        out[target_col] = best_values
        print(f"[metadata] HudAlphaID를 '{best_name}'에서 추출/정규화했습니다. unique={out[target_col].nunique(dropna=True)}")
    else:
        print("[metadata] 경고: 5104-SL-... 형식의 HudAlphaID를 찾지 못했습니다.")

    return out


def get_gene_feature_cols(df: pd.DataFrame) -> List[str]:
    """TPM matrix에서 ENSG 또는 ENST로 시작하는 feature column을 추출합니다."""
    feature_cols = []
    for c in df.columns:
        if re.match(r"^(ENSG|ENST)\d+", str(c)):
            feature_cols.append(c)
    return feature_cols


def resolve_auto_column(
    df: pd.DataFrame,
    requested: str,
    candidates: Sequence[str],
    role: str,
    required: bool = True,
) -> Optional[str]:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"{role} column '{requested}'이 DataFrame에 없습니다.")
        return requested

    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(f"{role} column을 찾지 못했습니다. 후보: {candidates}")
    return None


def load_and_merge_ppmi_rna(tpm_path: Path, meta_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    기존 노트북과 동일한 방식으로 TPM matrix와 metadata를 SAMPLE_ID_norm 기준으로 병합합니다.
    """
    tpm_path = Path(tpm_path)
    meta_path = Path(meta_path)

    if not tpm_path.exists():
        raise FileNotFoundError(f"TPM matrix를 찾을 수 없습니다: {tpm_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata를 찾을 수 없습니다: {meta_path}")

    print(f"[load] TPM: {tpm_path}")
    print(f"[load] metadata: {meta_path}")

    tpm_df = pd.read_csv(tpm_path, index_col=0, low_memory=False)
    meta_df = pd.read_csv(meta_path, index_col=0, low_memory=False)

    tpm_df = tpm_df.copy()
    meta_df = meta_df.copy()

    # TPM side
    tpm_df["PATNO_norm"] = tpm_df["PATNO"].apply(normalize_patno) if "PATNO" in tpm_df.columns else np.nan
    tpm_df["VISIT_norm"] = tpm_df["VISIT"].apply(normalize_visit) if "VISIT" in tpm_df.columns else np.nan
    tpm_df["SAMPLE_ID_norm"] = tpm_df["SAMPLE_ID"].apply(normalize_sample_id) if "SAMPLE_ID" in tpm_df.columns else np.nan

    # metadata side
    meta_df["PATNO_norm"] = meta_df["PATNO"].apply(normalize_patno) if "PATNO" in meta_df.columns else np.nan
    meta_df["VISIT_norm"] = meta_df["CLINICAL_EVENT"].apply(normalize_visit) if "CLINICAL_EVENT" in meta_df.columns else np.nan
    meta_df["SAMPLE_ID_norm"] = (
        meta_df["Specimen Bar Code"].apply(normalize_sample_id)
        if "Specimen Bar Code" in meta_df.columns
        else np.nan
    )

    meta_df = ensure_hudalphaid_column(meta_df, target_col="HudAlphaID")

    meta_keep_cols = [
        "SAMPLE_ID_norm",
        "PATNO_norm",
        "VISIT_norm",
        "HudAlphaID",
        "QCflagIR3",
        "DIAGNOSIS",
        "GENDER",
        "CLINICAL_EVENT",
    ]
    meta_keep_cols = [c for c in meta_keep_cols if c in meta_df.columns]
    meta_sub = meta_df[meta_keep_cols].copy()

    merged_df = pd.merge(
        tpm_df,
        meta_sub,
        on="SAMPLE_ID_norm",
        how="left",
        suffixes=("", "_meta"),
    )

    feature_cols = get_gene_feature_cols(merged_df)
    if len(feature_cols) == 0:
        raise ValueError("ENSG 또는 ENST로 시작하는 유전자 feature column을 찾지 못했습니다.")

    print("[merge] merged_df:", merged_df.shape)
    if "DIAGNOSIS" in merged_df.columns:
        print("[merge] DIAGNOSIS matched ratio:", float(merged_df["DIAGNOSIS"].notna().mean()))
    if "HudAlphaID" in merged_df.columns:
        print("[merge] HudAlphaID matched ratio:", float(merged_df["HudAlphaID"].notna().mean()))
    print("[merge] feature count:", len(feature_cols))

    return merged_df, feature_cols


def make_label_map(selected_groups: Sequence[str]) -> Dict[str, int]:
    preferred_order = ["Control", "PD", "Prodromal", "SWEDD"]
    ordered = []
    for g in preferred_order:
        if g in selected_groups:
            ordered.append(g)
    for g in selected_groups:
        if g not in ordered:
            ordered.append(g)
    return {g: i for i, g in enumerate(ordered)}


def build_filtered_df(
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    selected_groups: Sequence[str],
    visit_values: Optional[Sequence[str]],
    patient_col: str,
    diagnosis_col: str = "auto",
    visit_col: str = "auto",
    drop_qc_fail: bool = True,
    qc_col: str = "QCflagIR3",
    qc_pass_value: str = "pass",
    label_map: Optional[Dict[str, int]] = None,
    require_labels: bool = True,
    return_features: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[np.ndarray], Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    """
    selected_groups에 해당하는 sample을 필터링합니다.

    메모리 절약을 위해 먼저 boolean mask를 만든 뒤 필요한 행만 copy하고,
    반환용 metadata에서는 gene feature column을 제거합니다.
    return_features=False이면 X_df 생성을 건너뛰어 target split 전용 metadata만 만듭니다.
    """
    df0 = merged_df

    diagnosis_col_resolved = resolve_auto_column(
        df0,
        diagnosis_col,
        candidates=["DIAGNOSIS", "DIAGONOSIS", "DIAGNOSIS_STD", "diagnosis", "Diagnosis"],
        role="diagnosis",
        required=True,
    )

    visit_col_resolved = resolve_auto_column(
        df0,
        visit_col,
        candidates=["VISIT_final", "CLINICAL_EVENT", "VISIT", "EVENT_ID", "VISIT_norm"],
        role="visit",
        required=(visit_values is not None),
    )

    if patient_col not in df0.columns:
        raise ValueError(f"patient_col='{patient_col}'이 merged_df에 없습니다. 사용 가능한 컬럼 예: {list(df0.columns[:20])}")

    diagnosis_group = df0[diagnosis_col_resolved].apply(normalize_diagnosis_for_tabnet)
    selected_groups = list(selected_groups)

    mask = diagnosis_group.isin(selected_groups)
    mask &= df0[patient_col].notna()

    if visit_values is not None and len(visit_values) > 0:
        visit_values_norm = [normalize_visit_for_filter(v) for v in visit_values]
        visit_for_filter = df0[visit_col_resolved].apply(normalize_visit_for_filter)
        mask &= visit_for_filter.isin(visit_values_norm)

    if drop_qc_fail and qc_col in df0.columns:
        mask &= df0[qc_col].eq(qc_pass_value)

    df = df0.loc[mask].copy()
    df["DIAGNOSIS_GROUP"] = diagnosis_group.loc[mask].to_numpy()
    df[patient_col] = df[patient_col].astype(str)

    y = None
    inverse_label_map = None

    if require_labels:
        if label_map is None:
            label_map = make_label_map(selected_groups)

        df["label"] = df["DIAGNOSIS_GROUP"].map(label_map)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        y = df["label"].to_numpy(dtype=np.int64)
        inverse_label_map = {v: k for k, v in label_map.items()}

    X_df = None
    if return_features:
        # 빠른 경로: TPM columns가 이미 numeric이면 바로 float32 numpy로 변환합니다.
        # 예외가 발생할 때만 pd.to_numeric(errors='coerce') fallback을 사용합니다.
        try:
            X_np = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        except Exception:
            X_tmp = df[feature_cols].apply(pd.to_numeric, errors="coerce")
            X_np = X_tmp.to_numpy(dtype=np.float32, copy=True)
            del X_tmp

        np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(X_np, 0.0, out=X_np)
        np.log1p(X_np, out=X_np)
        X_df = pd.DataFrame(X_np, columns=feature_cols, index=df.index).reset_index(drop=True)

    # metadata 용도로 쓰는 df에는 gene feature를 남기지 않습니다.
    meta_df = df.drop(columns=feature_cols, errors="ignore").reset_index(drop=True)

    return meta_df, X_df, y, label_map, inverse_label_map


# ---------------------------------------------------------------------
# Feature selection, CV splitting
# ---------------------------------------------------------------------

def get_cv_splitter(n_splits: int, random_state: int):
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    warnings.warn(
        "StratifiedGroupKFold를 사용할 수 없어 GroupKFold로 대체합니다. "
        "class 비율이 fold별로 완전히 보존되지 않을 수 있습니다."
    )
    return GroupKFold(n_splits=n_splits)


def split_unsupervised_train_valid(
    df: pd.DataFrame,
    patient_col: str,
    valid_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    groups = df[patient_col].astype(str).to_numpy()

    if len(df) < 2 or len(np.unique(groups)) < 2 or valid_size <= 0:
        return idx, np.array([], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(idx, groups=groups))
    return train_idx, valid_idx


def select_feature_columns(
    X_train_df: pd.DataFrame,
    mode: str,
    top_n: Optional[int],
    variance_threshold: float = 0.0,
) -> List[str]:
    mode = str(mode).lower()

    if mode == "all":
        return list(X_train_df.columns)

    variances = X_train_df.var(axis=0, ddof=0).sort_values(ascending=False)

    if mode == "variance":
        if top_n is None or top_n <= 0 or top_n >= len(variances):
            selected = variances.index.tolist()
        else:
            selected = variances.head(top_n).index.tolist()

    elif mode == "variance_threshold":
        selected = variances[variances > variance_threshold].index.tolist()
        if top_n is not None and top_n > 0:
            selected = selected[:top_n]

    else:
        raise ValueError(f"Unknown feature_filter mode: {mode}")

    if len(selected) == 0:
        raise ValueError(
            f"feature_filter={mode} 결과 선택된 feature가 없습니다. "
            "top_n 또는 variance_threshold를 조정하세요."
        )

    return selected



# ---------------------------------------------------------------------
# TabNet model helpers
# ---------------------------------------------------------------------

def import_tabnet():
    try:
        import torch
        from pytorch_tabnet.pretraining import TabNetPretrainer
    except Exception as e:
        raise ImportError(
            "pytorch-tabnet 또는 torch import에 실패했습니다. "
            "설치 예: pip install pytorch-tabnet torch"
        ) from e
    return torch, TabNetPretrainer


def make_tabnet_params(args, seed: int):
    torch, _ = import_tabnet()

    return dict(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_sparse=args.lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=args.lr),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=args.scheduler_step_size, gamma=args.scheduler_gamma),
        mask_type=args.mask_type,
        seed=seed,
        device_name=args.device_name,
        verbose=args.verbose,
    )


def fit_pretrainer(
    X_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    args,
    seed: int,
):
    _, TabNetPretrainer = import_tabnet()

    params = make_tabnet_params(args, seed)
    model = TabNetPretrainer(**params)

    eval_set = [X_valid] if X_valid is not None and len(X_valid) > 0 else []

    model.fit(
        X_train=X_train,
        eval_set=eval_set,
        max_epochs=args.pretrain_max_epochs,
        patience=args.pretrain_patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pretraining_ratio=args.pretraining_ratio,
    )
    return model



def extract_encoder_latent(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """
    TabNetPretrainer 또는 TabNetClassifier에서 encoder latent z=sum(step_outputs)를 추출합니다.
    pytorch-tabnet 버전에 따라 network 구조가 다를 수 있어 가능한 경로를 순차적으로 탐색합니다.
    """
    torch, _ = import_tabnet()

    network = model.network
    network.eval()

    # network 구조 후보
    if hasattr(network, "embedder") and hasattr(network, "encoder"):
        embedder = network.embedder
        encoder = network.encoder
    elif hasattr(network, "embedder") and hasattr(network, "tabnet") and hasattr(network.tabnet, "encoder"):
        embedder = network.embedder
        encoder = network.tabnet.encoder
    elif hasattr(network, "tabnet") and hasattr(network.tabnet, "embedder") and hasattr(network.tabnet, "encoder"):
        embedder = network.tabnet.embedder
        encoder = network.tabnet.encoder
    else:
        raise AttributeError("model.network에서 embedder/encoder를 찾지 못했습니다.")

    device = next(network.parameters()).device

    z_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            x_emb = embedder(xb) if embedder is not None else xb
            steps_output, _ = encoder(x_emb)
            z = sum(steps_output)
            z_list.append(z.detach().cpu().numpy())

    return np.concatenate(z_list, axis=0)


def get_tabnet_encoder_parts(model):
    """pytorch-tabnet 버전 차이를 고려해 network/embedder/encoder를 찾습니다."""
    network = model.network

    if hasattr(network, "embedder") and hasattr(network, "encoder"):
        return network, network.embedder, network.encoder
    if hasattr(network, "embedder") and hasattr(network, "tabnet") and hasattr(network.tabnet, "encoder"):
        return network, network.embedder, network.tabnet.encoder
    if hasattr(network, "tabnet") and hasattr(network.tabnet, "embedder") and hasattr(network.tabnet, "encoder"):
        return network, network.tabnet.embedder, network.tabnet.encoder

    raise AttributeError("model.network에서 embedder/encoder를 찾지 못했습니다.")


def state_dict_to_cpu(state_dict):
    """state_dict tensor를 CPU로 옮겨 저장 호환성을 높입니다."""
    torch, _ = import_tabnet()
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def save_pretrained_encoder_artifacts(
    pretrainer,
    fold_dir: Path,
    selected_feature_cols: Sequence[str],
    args,
    fold_summary: Dict,
    scaler_path: Optional[Path] = None,
) -> None:
    """
    나중에 재사용할 수 있도록 Encoder 중심 산출물을 저장합니다.

    - tabnet_pretrainer_full.zip은 pytorch-tabnet의 load_model / from_unsupervised 용도에 가장 안전합니다.
    - *_state_dict.pt 파일들은 직접 PyTorch 모듈 수준에서 가중치를 다루고자 할 때 사용합니다.
    - bundle에는 feature 순서와 scaler 경로까지 함께 저장합니다.
    """
    torch, _ = import_tabnet()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    network, embedder, encoder = get_tabnet_encoder_parts(pretrainer)

    encoder_sd = state_dict_to_cpu(encoder.state_dict())
    network_sd = state_dict_to_cpu(network.state_dict())
    embedder_sd = None
    if embedder is not None and hasattr(embedder, "state_dict"):
        embedder_sd = state_dict_to_cpu(embedder.state_dict())

    torch.save(encoder_sd, fold_dir / "encoder_state_dict.pt")
    if getattr(args, "save_network_state_dict", False):
        torch.save(network_sd, fold_dir / "network_state_dict.pt")
    else:
        del network_sd
    if embedder_sd is not None:
        torch.save(embedder_sd, fold_dir / "embedder_state_dict.pt")

    bundle = {
        "encoder_state_dict": encoder_sd,
        "embedder_state_dict": embedder_sd,
        "selected_feature_cols": list(selected_feature_cols),
        "n_features_selected": int(len(selected_feature_cols)),
        "scaler_path": str(scaler_path) if scaler_path is not None else None,
        "tabnet_params": {
            "n_d": args.n_d,
            "n_a": args.n_a,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "lambda_sparse": args.lambda_sparse,
            "mask_type": args.mask_type,
            "pretraining_ratio": args.pretraining_ratio,
        },
        "fold_summary": safe_to_jsonable(fold_summary),
        "note": (
            "Downstream에서 사용할 때는 selected_feature_cols 순서로 X를 구성하고, "
            "scaler_path가 None이 아니면 동일 scaler를 적용해야 합니다. "
            "pytorch-tabnet의 from_unsupervised에는 tabnet_pretrainer_full.zip을 load_model로 불러와 사용하는 편이 가장 안전합니다."
        ),
    }
    torch.save(bundle, fold_dir / "pretrained_encoder_bundle.pt")

    usage_text = f"""Pretrained TabNet Encoder 사용 메모
=================================

이 fold의 저장 파일:
- tabnet_pretrainer_full.zip
- pretrained_encoder_bundle.pt
- encoder_state_dict.pt
- network_state_dict.pt (--save-network-state-dict 사용 시)
- selected_feature_columns.csv
- {'scaler.joblib' if scaler_path is not None else 'scaler 없음 (--no-scale-within-fold)'}

권장 사용 방식:
1) selected_feature_columns.csv의 feature 순서대로 downstream X를 구성합니다.
2) scaler.joblib가 있으면 동일 scaler로 transform합니다.
3) pytorch-tabnet에서 fine-tuning할 경우 TabNetPretrainer를 생성한 뒤
   load_model('tabnet_pretrainer_full.zip')로 불러오고,
   TabNetClassifier.fit(..., from_unsupervised=loaded_pretrainer)를 사용합니다.

주의:
- fold-wise pretraining이므로 fold별 validation subject는 해당 fold의 pretraining에 포함되지 않았습니다.
- 다른 fold의 downstream split에 사용할 때는 누수 여부를 다시 확인해야 합니다.
"""
    (fold_dir / "README_pretrained_encoder_usage.txt").write_text(usage_text, encoding="utf-8")


# ---------------------------------------------------------------------
# Main CV experiment
# ---------------------------------------------------------------------

def run_fold_experiment(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    target_df: pd.DataFrame,
    target_X_df: Optional[pd.DataFrame],
    target_y: np.ndarray,
    pretrain_df: pd.DataFrame,
    pretrain_X_df: pd.DataFrame,
    feature_cols: List[str],
    patient_col: str,
    args,
    run_dir: Path,
):

    seed = args.random_state

    fold_dir = run_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    target_train_df = target_df.iloc[train_idx].reset_index(drop=True)
    target_valid_df = target_df.iloc[valid_idx].reset_index(drop=True)

    valid_subjects = set(target_valid_df[patient_col].astype(str).tolist())
    train_subjects = set(target_train_df[patient_col].astype(str).tolist())
    overlap = train_subjects & valid_subjects
    if overlap:
        raise RuntimeError(f"Fold {fold}: target train/valid subject overlap 발생: {len(overlap)}")

    # validation/test subject는 self-supervised pretraining에서도 제외합니다.
    pretrain_pool_mask = ~pretrain_df[patient_col].astype(str).isin(valid_subjects)
    pretrain_pool_df = pretrain_df.loc[pretrain_pool_mask].reset_index(drop=True)
    pretrain_pool_X_df = pretrain_X_df.iloc[pretrain_pool_mask.to_numpy()].reset_index(drop=True)

    unsup_train_idx, unsup_valid_idx = split_unsupervised_train_valid(
        pretrain_pool_df,
        patient_col=patient_col,
        valid_size=args.pretrain_valid_size,
        random_state=seed,
    )

    X_unsup_train_full = pretrain_pool_X_df.iloc[unsup_train_idx].reset_index(drop=True)
    X_unsup_valid_full = (
        pretrain_pool_X_df.iloc[unsup_valid_idx].reset_index(drop=True)
        if len(unsup_valid_idx) > 0
        else pd.DataFrame(columns=pretrain_pool_X_df.columns)
    )

    selected_feature_cols = select_feature_columns(
        X_train_df=X_unsup_train_full,
        mode=args.feature_filter,
        top_n=args.top_n_features,
        variance_threshold=args.variance_threshold,
    )

    pd.DataFrame({"feature": selected_feature_cols}).to_csv(
        fold_dir / "selected_feature_columns.csv",
        index=False,
    )

    X_unsup_train_df = X_unsup_train_full[selected_feature_cols]
    X_unsup_valid_df = X_unsup_valid_full[selected_feature_cols] if len(unsup_valid_idx) > 0 else None

    scaler = None
    scaler_path = None
    if args.scale_within_fold:
        scaler = StandardScaler()
        X_unsup_train = scaler.fit_transform(X_unsup_train_df).astype(np.float32, copy=False)
        X_unsup_valid = (
            scaler.transform(X_unsup_valid_df).astype(np.float32, copy=False)
            if X_unsup_valid_df is not None and len(X_unsup_valid_df) > 0
            else None
        )
        scaler_path = fold_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
    else:
        X_unsup_train = X_unsup_train_df.to_numpy(dtype=np.float32, copy=True)
        X_unsup_valid = (
            X_unsup_valid_df.to_numpy(dtype=np.float32, copy=True)
            if X_unsup_valid_df is not None and len(X_unsup_valid_df) > 0
            else None
        )

    X_unsup_train = make_contiguous_float32(X_unsup_train)
    X_unsup_valid = make_contiguous_float32(X_unsup_valid)

    fold_summary = {
        "fold": int(fold),
        "seed": int(seed),
        "n_features_selected": int(len(selected_feature_cols)),
        "target_train_samples": int(len(train_idx)),
        "target_valid_samples": int(len(valid_idx)),
        "target_train_subjects": int(len(train_subjects)),
        "target_valid_subjects": int(len(valid_subjects)),
        "pretrain_pool_samples": int(len(pretrain_pool_df)),
        "pretrain_pool_subjects": int(pretrain_pool_df[patient_col].nunique()),
        "pretrain_train_samples": int(len(X_unsup_train)),
        "pretrain_valid_samples": int(0 if X_unsup_valid is None else len(X_unsup_valid)),
        "pretrain_groups_distribution": pretrain_pool_df["DIAGNOSIS_GROUP"].value_counts().to_dict(),
        "target_train_distribution": target_train_df["DIAGNOSIS_GROUP"].value_counts().to_dict(),
        "target_valid_distribution": target_valid_df["DIAGNOSIS_GROUP"].value_counts().to_dict(),
        "scaler_path": str(scaler_path) if scaler_path is not None else None,
    }
    save_json(fold_summary, fold_dir / "fold_summary.json")

    print("=" * 100)
    print(f"[Fold {fold}] selected features: {len(selected_feature_cols)}")
    print(f"[Fold {fold}] pretrain train/valid: {len(X_unsup_train)} / {0 if X_unsup_valid is None else len(X_unsup_valid)}")
    print(f"[Fold {fold}] target train/valid subjects: {len(train_subjects)} / {len(valid_subjects)}")
    print(f"[Fold {fold}] TabNetPretrainer self-supervised pretraining only")

    pretrainer = fit_pretrainer(
        X_train=X_unsup_train,
        X_valid=X_unsup_valid,
        args=args,
        seed=seed,
    )

    # 1) pytorch-tabnet 전체 pretrainer 저장: from_unsupervised 재사용에 가장 안전합니다.
    pretrainer.save_model(str(fold_dir / "tabnet_pretrainer_full"))
    save_history(pretrainer, fold_dir / "pretrainer_history.csv")

    # 2) Encoder 중심 state_dict bundle 저장
    save_pretrained_encoder_artifacts(
        pretrainer=pretrainer,
        fold_dir=fold_dir,
        selected_feature_cols=selected_feature_cols,
        args=args,
        fold_summary=fold_summary,
        scaler_path=scaler_path,
    )

    # 3) 선택적으로 latent representation 저장. 기본값은 False입니다.
    if args.extract_latent:
        if target_X_df is None:
            warnings.warn("target_X_df가 없어 latent extraction을 건너뜁니다. return_features 설정을 확인하세요.")
        else:
            try:
                X_target_train_df = target_X_df.iloc[train_idx].reset_index(drop=True)[selected_feature_cols]
                X_target_valid_df = target_X_df.iloc[valid_idx].reset_index(drop=True)[selected_feature_cols]
                if scaler is not None:
                    X_target_train = scaler.transform(X_target_train_df).astype(np.float32, copy=False)
                    X_target_valid = scaler.transform(X_target_valid_df).astype(np.float32, copy=False)
                else:
                    X_target_train = X_target_train_df.to_numpy(dtype=np.float32, copy=True)
                    X_target_valid = X_target_valid_df.to_numpy(dtype=np.float32, copy=True)
                X_target_train = make_contiguous_float32(X_target_train)
                X_target_valid = make_contiguous_float32(X_target_valid)

                z_target_train = extract_encoder_latent(pretrainer, X_target_train, batch_size=args.infer_batch_size)
                z_target_valid = extract_encoder_latent(pretrainer, X_target_valid, batch_size=args.infer_batch_size)
                np.save(fold_dir / "latent_target_train_pretrained_encoder.npy", z_target_train)
                np.save(fold_dir / "latent_target_valid_pretrained_encoder.npy", z_target_valid)
                target_train_df.to_csv(fold_dir / "latent_target_train_metadata.csv", index=False)
                target_valid_df.to_csv(fold_dir / "latent_target_valid_metadata.csv", index=False)

                del X_target_train_df, X_target_valid_df, X_target_train, X_target_valid
                del z_target_train, z_target_valid
            except Exception as e:
                warnings.warn(f"Fold {fold}: latent extraction 실패: {e}")

    del pretrainer
    del X_unsup_train_full, X_unsup_valid_full, X_unsup_train_df, X_unsup_valid_df
    del X_unsup_train, X_unsup_valid, pretrain_pool_df, pretrain_pool_X_df
    del target_train_df, target_valid_df
    clear_cuda()

    return fold_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPMI RNA-seq TabNet Encoder self-supervised pretraining only."
    )

    # data
    parser.add_argument("--tpm-path", type=Path, default=Path("./Dataset/RNA/IR3/quant/tpm_gene_matrix.csv"))
    parser.add_argument("--meta-path", type=Path, default=Path("./Dataset/RNA/IR3/metaDataIR3.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("./results_tabnet_rna_ssl_pretrain_only"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--pretrain-groups", nargs="+", default=["PD", "Control", "Prodromal", "SWEDD", "Genetic Cohort", "Genetic Registry"])
    parser.add_argument("--target-groups", nargs="+", default=["PD", "Control"])
    parser.add_argument("--visit-values", nargs="*", default=None)
    parser.add_argument("--patient-col", type=str, default="PATNO_norm")
    parser.add_argument("--diagnosis-col", type=str, default="auto")
    parser.add_argument("--visit-col", type=str, default="auto")
    parser.add_argument("--drop-qc-fail", action=argparse.BooleanOptionalAction, default=True)

    # CV / fold control
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--only-folds", nargs="*", type=int, default=None,
                        help="지정한 fold 번호만 실행합니다. 예: --only-folds 1 3 5")
    parser.add_argument("--pretrain-valid-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # feature filtering
    parser.add_argument("--feature-filter", type=str, default="variance", choices=["all", "variance", "variance_threshold"])
    parser.add_argument("--top-n-features", type=int, default=5000)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--scale-within-fold", action=argparse.BooleanOptionalAction, default=True)

    # TabNet architecture
    parser.add_argument("--n-d", type=int, default=4)
    parser.add_argument("--n-a", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.3)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--mask-type", type=str, default="sparsemax", choices=["sparsemax", "entmax"])

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--virtual-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-network-state-dict", action=argparse.BooleanOptionalAction, default=False,
                        help="Encoder 외에 pretrainer network 전체 state_dict도 별도로 저장합니다.")
    parser.add_argument("--device-name", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=1)

    # pretraining
    parser.add_argument("--pretraining-ratio", type=float, default=0.3)
    parser.add_argument("--pretrain-max-epochs", type=int, default=300)
    parser.add_argument("--pretrain-patience", type=int, default=100)

    # optional latent representation
    parser.add_argument("--extract-latent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--infer-batch-size", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)

    run_name = args.run_name
    if run_name is None:
        run_name = f"ppmi_tabnet_ssl_pretrain_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(safe_to_jsonable(vars(args)), run_dir / "config.json")

    print("[run_dir]", run_dir.resolve())

    # Load and prepare data
    merged_df, feature_cols = load_and_merge_ppmi_rna(args.tpm_path, args.meta_path)

    # Target labels are used only to define leak-safe subject-level folds.
    target_label_map = make_label_map(args.target_groups)
    target_df, target_X_df, target_y, label_map, inverse_label_map = build_filtered_df(
        merged_df=merged_df,
        feature_cols=feature_cols,
        selected_groups=args.target_groups,
        visit_values=args.visit_values,
        patient_col=args.patient_col,
        diagnosis_col=args.diagnosis_col,
        visit_col=args.visit_col,
        drop_qc_fail=args.drop_qc_fail,
        label_map=target_label_map,
        require_labels=True,
        return_features=args.extract_latent,
    )

    # Pretraining data: labels are not used, but group distribution is kept for reporting.
    pretrain_df, pretrain_X_df, _, _, _ = build_filtered_df(
        merged_df=merged_df,
        feature_cols=feature_cols,
        selected_groups=args.pretrain_groups,
        visit_values=args.visit_values,
        patient_col=args.patient_col,
        diagnosis_col=args.diagnosis_col,
        visit_col=args.visit_col,
        drop_qc_fail=args.drop_qc_fail,
        label_map=None,
        require_labels=False,
        return_features=True,
    )

    # merged_df는 이후 필요 없으므로 즉시 해제합니다.
    del merged_df
    clear_cuda()

    class_names = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]

    dataset_summary = {
        "n_features_total": int(len(feature_cols)),
        "target_groups": list(args.target_groups),
        "pretrain_groups": list(args.pretrain_groups),
        "target_samples": int(len(target_df)),
        "target_subjects": int(target_df[args.patient_col].nunique()),
        "target_distribution": target_df["DIAGNOSIS_GROUP"].value_counts().to_dict(),
        "pretrain_samples": int(len(pretrain_df)),
        "pretrain_subjects": int(pretrain_df[args.patient_col].nunique()),
        "pretrain_distribution": pretrain_df["DIAGNOSIS_GROUP"].value_counts().to_dict(),
        "label_map": label_map,
        "class_names": class_names,
        "target_features_loaded": bool(target_X_df is not None),
    }
    save_json(dataset_summary, run_dir / "dataset_summary.json")
    pd.DataFrame({"feature": feature_cols}).to_csv(run_dir / "feature_columns_all.csv", index=False)
    pd.DataFrame(list(label_map.items()), columns=["group", "label"]).to_csv(run_dir / "label_map.csv", index=False)

    print("[dataset summary]")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    if len(target_df) == 0:
        raise ValueError("target_df가 비어 있습니다. --target-groups, --visit-values, metadata 병합 상태를 확인하세요.")
    if len(pretrain_df) == 0 or pretrain_X_df is None:
        raise ValueError("pretrain_df가 비어 있습니다. --pretrain-groups, --visit-values, metadata 병합 상태를 확인하세요.")

    groups_target = target_df[args.patient_col].astype(str).to_numpy()
    splitter = get_cv_splitter(args.n_splits, args.random_state)

    fold_summaries = []
    only_folds = set(args.only_folds) if args.only_folds else None
    dummy_X_for_split = np.zeros((len(target_df), 1), dtype=np.float32)
    split_iter = splitter.split(dummy_X_for_split, target_y, groups=groups_target)

    for fold, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        if only_folds is not None and fold not in only_folds:
            continue

        fold_summary = run_fold_experiment(
            fold=fold,
            train_idx=train_idx,
            valid_idx=valid_idx,
            target_df=target_df,
            target_X_df=target_X_df,
            target_y=target_y,
            pretrain_df=pretrain_df,
            pretrain_X_df=pretrain_X_df,
            feature_cols=feature_cols,
            patient_col=args.patient_col,
            args=args,
            run_dir=run_dir,
        )
        fold_summaries.append(fold_summary)

    if fold_summaries:
        pd.DataFrame(fold_summaries).to_csv(run_dir / "pretraining_fold_summaries.csv", index=False)
        print("[pretraining_fold_summaries]")
        print(pd.DataFrame(fold_summaries))
    else:
        print("[done] 실행된 fold가 없습니다. --only-folds 값을 확인하세요.")

    clear_cuda()
    print("[done] pretrained encoders saved to:", run_dir.resolve())


if __name__ == "__main__":
    main()
