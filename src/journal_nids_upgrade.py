"""Journal-ready hybrid NIDS experiment runner.

This module implements the NF-ToN-IoT-V2 -> XGBoost + Isolation Forest ->
NF-UNSW-NB15 protocol used for publication artifacts. It is intentionally
separate from the dashboard backend so the existing application remains stable
while the research experiment can be reproduced from the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.project_paths import DATA_DIR, MODEL_DIR, PAPER_DIR, RESULTS_DIR


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


warnings.filterwarnings("ignore")

SEED = 42
CONF_THR = 0.65
ALPHA = 0.6
UNKNOWN_LABEL = "Unknown Attack"
LABEL_COLUMNS = {"Label", "Attack", "Attack_Type"}

THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
MODEL_NAMES = {
    "xgb_multi": "XGBoost multiclass",
    "xgb_binary": "XGBoost binary",
    "iso": "Isolation Forest",
}


@dataclass
class ExperimentConfig:
    train_data: Path
    external_data: Path
    output_dir: Path
    figure_dir: Path
    table_dir: Path
    model_dir: Path
    max_train_rows: int | None = None
    max_external_rows: int | None = None
    sample_mode: str = "head"
    test_size: float = 0.20
    confidence_threshold: float = CONF_THR
    alpha: float = ALPHA
    shap_sample_size: int = 500
    skip_shap: bool = False
    xgb_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    iso_estimators: int = 100
    n_jobs: int = -1
    seed: int = SEED
    use_smote: bool = True


@dataclass
class PreparedData:
    feature_cols: list[str]
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    y_train_multi: pd.Series
    y_val_multi: pd.Series
    y_train_bin: pd.Series
    y_val_bin: pd.Series
    X_resampled: np.ndarray
    y_resampled_multi: pd.Series
    y_resampled_bin: np.ndarray
    X_binary_resampled: np.ndarray
    y_binary_resampled: np.ndarray
    scaler: StandardScaler
    class_distribution_before: dict[str, int]
    class_distribution_after: dict[str, int]
    smote_note: str
    binary_distribution_after: dict[str, int]
    binary_smote_note: str


@dataclass
class TrainedModels:
    xgb_multi: Any
    xgb_binary: Any
    iso: IsolationForest
    multi_encoder: LabelEncoder
    binary_encoder: LabelEncoder


def setup_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def default_train_data() -> Path:
    csv_path = DATA_DIR / "NF-ToN-IoT-V2.csv"
    parquet_path = DATA_DIR / "NF-ToN-IoT-V2.parquet"
    return csv_path if csv_path.exists() else parquet_path


def default_external_data() -> Path:
    preferred = DATA_DIR / "NF-UNSW-NB15-v3.csv"
    fallback = DATA_DIR / "NF-UNSW-NB15.csv"
    return preferred if preferred.exists() else fallback


def import_xgboost() -> Any:
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is required for the journal NIDS upgrade. "
            "Install dependencies with: venv\\Scripts\\pip install -r requirements.txt"
        ) from exc
    return xgb


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def ensure_dirs(config: ExperimentConfig) -> None:
    for path in (config.output_dir, config.figure_dir, config.table_dir, config.model_dir):
        path.mkdir(parents=True, exist_ok=True)


def log_step(step: int, title: str, payload: dict[str, Any] | None = None) -> None:
    print(f"[STEP {step}] {title}")
    if payload:
        for key, value in payload.items():
            print(f"  - {key}: {value}")


def read_flow_dataset(
    path: Path,
    max_rows: int | None = None,
    seed: int = SEED,
    sample_mode: str = "head",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        if max_rows is None:
            return pd.read_parquet(path)
        return read_parquet_head(path, max_rows)
    if max_rows is not None and sample_mode == "random":
        return read_csv_random_sample(path, max_rows, seed)
    return pd.read_csv(path, nrows=max_rows)


def read_csv_random_sample(path: Path, max_rows: int, seed: int, chunksize: int = 200_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        combined = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        reservoir = combined.nsmallest(max_rows, "_sample_key")

    if reservoir is None:
        return pd.DataFrame()
    sampled = reservoir.drop(columns=["_sample_key"]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def read_parquet_head(path: Path, max_rows: int) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        pieces = []
        remaining = max_rows
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group)
            frame = table.to_pandas()
            pieces.append(frame.head(remaining))
            remaining -= len(pieces[-1])
            if remaining <= 0:
                break
        return pd.concat(pieces, ignore_index=True)
    except Exception:
        return pd.read_parquet(path).head(max_rows)


def clean_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    missing_labels = {"Label", "Attack"} - set(df.columns)
    if missing_labels:
        raise ValueError(f"Training data is missing required columns: {sorted(missing_labels)}")

    cleaned = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    cleaned["Label"] = cleaned["Label"].astype(int)
    cleaned["Attack_Type"] = cleaned["Attack"].astype(str)

    feature_cols = [
        column
        for column in cleaned.columns
        if column not in LABEL_COLUMNS and pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns were found after cleaning.")

    X = cleaned[feature_cols].reset_index(drop=True)
    y_multi = cleaned["Attack_Type"].reset_index(drop=True)
    y_bin = cleaned["Label"].reset_index(drop=True)
    return X, y_multi, y_bin, feature_cols


def clean_external_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if "Label" not in df.columns:
        raise ValueError("External data is missing required column: Label")

    cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Label"]).copy()
    cleaned["Label"] = cleaned["Label"].astype(int)
    attack_type = cleaned["Attack"].astype(str) if "Attack" in cleaned.columns else pd.Series(["Unknown"] * len(cleaned))

    for column in feature_cols:
        if column not in cleaned.columns:
            cleaned[column] = 0

    X = cleaned[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
    y = cleaned["Label"].reset_index(drop=True)
    attack_type = attack_type.reset_index(drop=True)
    return X, y, attack_type


def binary_from_attack_types(labels: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    series = pd.Series(labels).astype(str).str.lower()
    benign_names = {"benign", "normal", "0"}
    return np.where(series.isin(benign_names), 0, 1)


def split_and_scale(
    X: pd.DataFrame,
    y_multi: pd.Series,
    y_bin: pd.Series,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, pd.Series, StandardScaler]:
    counts = y_multi.value_counts()
    stratify = y_multi if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_val, y_train_multi, y_val_multi, y_train_bin, y_val_bin = train_test_split(
        X,
        y_multi,
        y_bin,
        test_size=config.test_size,
        stratify=stratify,
        random_state=config.seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return (
        X_train_scaled,
        X_val_scaled,
        y_train_multi.reset_index(drop=True),
        y_val_multi.reset_index(drop=True),
        y_train_bin.reset_index(drop=True),
        y_val_bin.reset_index(drop=True),
        scaler,
    )


def resample_training_data(
    X_train_scaled: np.ndarray,
    y_train_multi: pd.Series,
    config: ExperimentConfig,
) -> tuple[np.ndarray, pd.Series, np.ndarray, dict[str, int], str]:
    before = y_train_multi.value_counts().sort_index().astype(int).to_dict()
    if not config.use_smote:
        y_bin = binary_from_attack_types(y_train_multi)
        return X_train_scaled, y_train_multi.reset_index(drop=True), y_bin, before, "SMOTETomek disabled by CLI flag."

    counts = y_train_multi.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        y_bin = binary_from_attack_types(y_train_multi)
        return (
            X_train_scaled,
            y_train_multi.reset_index(drop=True),
            y_bin,
            before,
            "SMOTETomek skipped because at least one class has fewer than two samples.",
        )

    k_neighbors = max(1, min(5, int(counts.min()) - 1))
    smote = SMOTE(random_state=config.seed, k_neighbors=k_neighbors)
    smt = SMOTETomek(random_state=config.seed, smote=smote)
    X_resampled, y_resampled = smt.fit_resample(X_train_scaled, y_train_multi)
    y_resampled = pd.Series(y_resampled, name="Attack_Type").reset_index(drop=True)
    y_bin = binary_from_attack_types(y_resampled)
    after = y_resampled.value_counts().sort_index().astype(int).to_dict()
    note = f"SMOTETomek applied with SMOTE k_neighbors={k_neighbors}."
    return X_resampled, y_resampled, y_bin, after, note


def resample_binary_training_data(
    X_train_scaled: np.ndarray,
    y_train_bin: pd.Series,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], str]:
    y_binary = np.asarray(y_train_bin, dtype=int)
    before = pd.Series(y_binary).value_counts().sort_index().astype(int).to_dict()
    if not config.use_smote:
        return X_train_scaled, y_binary, {str(key): value for key, value in before.items()}, "Binary SMOTETomek disabled by CLI flag."

    counts = pd.Series(y_binary).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return (
            X_train_scaled,
            y_binary,
            {str(key): value for key, value in before.items()},
            "Binary SMOTETomek skipped because at least one class has fewer than two samples.",
        )

    k_neighbors = max(1, min(5, int(counts.min()) - 1))
    smote = SMOTE(random_state=config.seed, k_neighbors=k_neighbors)
    smt = SMOTETomek(random_state=config.seed, smote=smote)
    X_resampled, y_resampled = smt.fit_resample(X_train_scaled, y_binary)
    after = pd.Series(y_resampled).value_counts().sort_index().astype(int).to_dict()
    note = f"Binary SMOTETomek applied with SMOTE k_neighbors={k_neighbors}."
    return X_resampled, np.asarray(y_resampled, dtype=int), {str(key): value for key, value in after.items()}, note


def prepare_data(config: ExperimentConfig) -> PreparedData:
    train_df = read_flow_dataset(config.train_data, config.max_train_rows, config.seed, config.sample_mode)
    X, y_multi, y_bin, feature_cols = clean_training_frame(train_df)
    (
        X_train_scaled,
        X_val_scaled,
        y_train_multi,
        y_val_multi,
        y_train_bin,
        y_val_bin,
        scaler,
    ) = split_and_scale(X, y_multi, y_bin, config)

    X_resampled, y_resampled_multi, y_resampled_bin, after, smote_note = resample_training_data(
        X_train_scaled,
        y_train_multi,
        config,
    )
    X_binary_resampled, y_binary_resampled, binary_after, binary_smote_note = resample_binary_training_data(
        X_train_scaled,
        y_train_bin,
        config,
    )

    return PreparedData(
        feature_cols=feature_cols,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        y_train_multi=y_train_multi,
        y_val_multi=y_val_multi,
        y_train_bin=y_train_bin,
        y_val_bin=y_val_bin,
        X_resampled=X_resampled,
        y_resampled_multi=y_resampled_multi,
        y_resampled_bin=y_resampled_bin,
        X_binary_resampled=X_binary_resampled,
        y_binary_resampled=y_binary_resampled,
        scaler=scaler,
        class_distribution_before=y_train_multi.value_counts().sort_index().astype(int).to_dict(),
        class_distribution_after=after,
        smote_note=smote_note,
        binary_distribution_after=binary_after,
        binary_smote_note=binary_smote_note,
    )


def train_xgb_classifier(
    X_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: np.ndarray,
    y_val: pd.Series | np.ndarray,
    config: ExperimentConfig,
) -> tuple[Any, LabelEncoder]:
    xgb = import_xgboost()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_series = pd.Series(y_val)
    known_mask = y_val_series.isin(encoder.classes_)
    eval_set = None
    if known_mask.any():
        eval_set = [(X_val[known_mask.to_numpy()], encoder.transform(y_val_series[known_mask]))]

    class_count = len(encoder.classes_)
    objective = "binary:logistic" if class_count == 2 else "multi:softprob"
    eval_metric = "logloss" if class_count == 2 else "mlogloss"
    params: dict[str, Any] = {
        "n_estimators": config.xgb_estimators,
        "max_depth": config.xgb_max_depth,
        "learning_rate": config.xgb_learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": objective,
        "eval_metric": eval_metric,
        "base_score": 0.5,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "tree_method": "hist",
    }
    if class_count > 2:
        params["num_class"] = class_count

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train_encoded, eval_set=eval_set, verbose=False)
    return model, encoder


def train_models(data: PreparedData, config: ExperimentConfig) -> TrainedModels:
    xgb_multi, multi_encoder = train_xgb_classifier(
        data.X_resampled,
        data.y_resampled_multi,
        data.X_val_scaled,
        data.y_val_multi,
        config,
    )
    xgb_binary, binary_encoder = train_xgb_classifier(
        data.X_binary_resampled,
        data.y_binary_resampled,
        data.X_val_scaled,
        data.y_val_bin,
        config,
    )
    iso = IsolationForest(
        n_estimators=config.iso_estimators,
        contamination=0.1,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    iso.fit(data.X_resampled)
    return TrainedModels(
        xgb_multi=xgb_multi,
        xgb_binary=xgb_binary,
        iso=iso,
        multi_encoder=multi_encoder,
        binary_encoder=binary_encoder,
    )


def predict_encoded(model: Any, encoder: LabelEncoder, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_scaled)
    predicted_encoded = np.argmax(proba, axis=1)
    predicted = encoder.inverse_transform(predicted_encoded)
    confidence = proba.max(axis=1)
    return predicted, confidence, proba


def apply_unknown_threshold(predicted: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    final_pred = np.asarray(predicted, dtype=object).copy()
    final_pred[confidence < threshold] = UNKNOWN_LABEL
    return final_pred


def normalize_anomaly_scores(raw_scores: np.ndarray, scaler: MinMaxScaler | None = None) -> tuple[np.ndarray, MinMaxScaler]:
    values = (-raw_scores).reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(values).flatten()
    else:
        normalized = scaler.transform(values).flatten()
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized, scaler


def get_severity(confidence: np.ndarray, ano_score: np.ndarray, alpha: float = ALPHA) -> tuple[pd.Series, np.ndarray]:
    score = alpha * confidence + (1 - alpha) * ano_score
    bins = [0, 0.40, 0.60, 0.80, 1.01]
    labels = ["Low", "Medium", "High", "Critical"]
    severity = pd.cut(score, bins=bins, labels=labels, right=False, include_lowest=True)
    return pd.Series(severity.astype(str)), score


def target_severity_from_attack(attack_labels: pd.Series | np.ndarray) -> pd.Series:
    targets = []
    for label in pd.Series(attack_labels).astype(str).str.lower():
        if label in {"benign", "normal", "0"}:
            targets.append("Low")
        elif any(token in label for token in ("ddos", "dos", "ransom", "backdoor", "exploit", "shellcode")):
            targets.append("Critical")
        elif any(token in label for token in ("scan", "recon", "password", "fuzzer", "injection", "xss")):
            targets.append("High")
        else:
            targets.append("Medium")
    return pd.Series(targets)


def predictions_to_binary(predictions: np.ndarray) -> np.ndarray:
    series = pd.Series(predictions).astype(str).str.lower()
    return np.where(series.isin({"benign", "normal", "0"}), 0, 1)


def safe_roc_auc_binary(y_true: np.ndarray | pd.Series, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def safe_roc_auc_multiclass(y_true: pd.Series, proba: np.ndarray, encoder: LabelEncoder) -> float:
    try:
        if len(encoder.classes_) < 2 or len(pd.Series(y_true).unique()) < 2:
            return float("nan")
        y_encoded = encoder.transform(y_true)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_encoded, proba[:, 1]))
        return float(roc_auc_score(y_encoded, proba, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def multiclass_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    encoder: LabelEncoder | None = None,
) -> dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if proba is not None and encoder is not None:
        metrics["AUC_ROC"] = safe_roc_auc_multiclass(y_true, proba, encoder)
    return metrics


def binary_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if score is not None:
        metrics["AUC_ROC"] = safe_roc_auc_binary(y_true, score)
    return metrics


def threshold_sensitivity(
    y_val: pd.Series,
    predicted: np.ndarray,
    confidence: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        final = apply_unknown_threshold(predicted, confidence, threshold)
        unknown_mask = final == UNKNOWN_LABEL
        rows.append(
            {
                "Threshold": threshold,
                "Unknown_pct": float(unknown_mask.mean() * 100),
                "F1_macro": float(f1_score(y_val, final, average="macro", zero_division=0)),
                "FP_unknown_rate": float(unknown_mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def infer_positive_probability(model: Any, encoder: LabelEncoder, proba: np.ndarray) -> np.ndarray:
    decoded_classes = encoder.inverse_transform(np.arange(len(encoder.classes_)))
    class_list = list(decoded_classes)
    positive_index = class_list.index(1) if 1 in class_list else len(class_list) - 1
    return proba[:, positive_index]


def evaluate_external_dataset(
    data: PreparedData,
    models: TrainedModels,
    anomaly_scaler: MinMaxScaler,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    external_df = read_flow_dataset(config.external_data, config.max_external_rows, config.seed, config.sample_mode)
    X_ext, y_ext, attack_type = clean_external_frame(external_df, data.feature_cols)
    X_ext_scaled = data.scaler.transform(X_ext)

    ext_pred, ext_conf, ext_proba = predict_encoded(models.xgb_binary, models.binary_encoder, X_ext_scaled)
    ext_final = np.asarray(ext_pred, dtype=object)
    ext_final[ext_conf < config.confidence_threshold] = UNKNOWN_LABEL
    ext_pred_binary = np.where(pd.Series(ext_final).astype(str).eq("0"), 0, predictions_to_binary(ext_final))
    ext_score = infer_positive_probability(models.xgb_binary, models.binary_encoder, ext_proba)
    ext_metrics = binary_metrics(y_ext, ext_pred_binary, ext_score)

    raw_ext = models.iso.decision_function(X_ext_scaled)
    ext_ano_score, _ = normalize_anomaly_scores(raw_ext, anomaly_scaler)
    ext_severity, ext_severity_score = get_severity(ext_conf, ext_ano_score, config.alpha)

    report = classification_report(y_ext, ext_pred_binary, zero_division=0, output_dict=True)
    summary = {
        "rows": int(len(y_ext)),
        "attack_distribution": attack_type.value_counts().astype(int).to_dict(),
        "unknown_pct": float((ext_final == UNKNOWN_LABEL).mean() * 100),
        "classification_report": report,
        "metrics": ext_metrics,
        "mean_severity_score": float(np.mean(ext_severity_score)),
        "severity_distribution": ext_severity.value_counts().sort_index().astype(int).to_dict(),
    }

    table = pd.DataFrame([{"Dataset": "NF-UNSW-NB15", **ext_metrics, "Unknown_pct": summary["unknown_pct"]}])
    return table, summary


def save_models(data: PreparedData, models: TrainedModels, config: ExperimentConfig) -> dict[str, str]:
    paths = {
        "xgb_model": config.model_dir / "xgb_model.pkl",
        "xgb_multiclass_model": config.model_dir / "xgb_multiclass_model.pkl",
        "xgb_binary_model": config.model_dir / "xgb_binary_model.pkl",
        "iso_model": config.model_dir / "iso_model.pkl",
        "scaler": config.model_dir / "scaler.pkl",
        "multi_label_encoder": config.model_dir / "multi_label_encoder.pkl",
        "binary_label_encoder": config.model_dir / "binary_label_encoder.pkl",
        "feature_columns": config.model_dir / "feature_columns.json",
    }
    joblib.dump(models.xgb_multi, paths["xgb_model"])
    joblib.dump(models.xgb_multi, paths["xgb_multiclass_model"])
    joblib.dump(models.xgb_binary, paths["xgb_binary_model"])
    joblib.dump(models.iso, paths["iso_model"])
    joblib.dump(data.scaler, paths["scaler"])
    joblib.dump(models.multi_encoder, paths["multi_label_encoder"])
    joblib.dump(models.binary_encoder, paths["binary_label_encoder"])
    paths["feature_columns"].write_text(json.dumps(data.feature_cols, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def save_table(df: pd.DataFrame, name: str, config: ExperimentConfig) -> dict[str, str]:
    csv_path = config.table_dir / f"{name}.csv"
    tex_path = config.table_dir / f"{name}.tex"
    md_path = config.table_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=lambda value: f"{value:.4f}")
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    return {"csv": str(csv_path), "tex": str(tex_path), "md": str(md_path)}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    def render_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    rows = [[render_cell(value) for value in row] for row in df.to_numpy()]
    headers = [str(column) for column in df.columns]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(headers[index])
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def save_figure(fig: plt.Figure, name: str, config: ExperimentConfig) -> dict[str, str]:
    png_path = config.figure_dir / f"{name}.png"
    pdf_path = config.figure_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, config: ExperimentConfig) -> dict[str, str]:
    labels = sorted(set(pd.Series(y_true).astype(str)) | set(pd.Series(y_pred).astype(str)))
    cm = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), max(5, len(labels) * 0.45)))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Recall-normalized"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    return save_figure(fig, "fig_01_confusion_matrix", config)


def per_class_f1_frame(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    rows = []
    for label, stats in report.items():
        if not isinstance(stats, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append({"Class": label, "F1": float(stats["f1-score"]), "Support": int(stats["support"])})
    return pd.DataFrame(rows).sort_values("F1", ascending=False)


def plot_per_class_f1(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(max(7, len(frame) * 0.55), 4.5))
    sns.barplot(data=frame, x="Class", y="F1", ax=ax, color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class F1")
    ax.set_xlabel("Attack class")
    ax.set_ylabel("F1-score")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, "fig_02_per_class_f1", config)


def model_feature_importance(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(feature_cols)))
    return pd.DataFrame({"Feature": feature_cols, "Mean_abs_SHAP": values}).sort_values("Mean_abs_SHAP", ascending=False)


def shap_values_by_class(shap_values: Any, feature_count: int) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(values) for values in shap_values]
    values = np.asarray(shap_values)
    if values.ndim == 2:
        return [values]
    if values.ndim == 3 and values.shape[1] == feature_count:
        return [values[:, :, index] for index in range(values.shape[2])]
    if values.ndim == 3 and values.shape[2] == feature_count:
        return [values[:, index, :] for index in range(values.shape[1])]
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def expected_value_for_class(expected_value: Any, class_index: int) -> float:
    if isinstance(expected_value, str):
        expected_value = json.loads(expected_value)
    values = np.asarray(expected_value, dtype=float)
    if values.ndim == 0:
        return float(values)
    return float(values[class_index])


def generate_shap_outputs(
    data: PreparedData,
    models: TrainedModels,
    final_predictions: np.ndarray,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str], dict[str, list[str]]]:
    warning_path = config.output_dir / "shap_warning.txt"
    if warning_path.exists():
        warning_path.unlink()

    if config.skip_shap:
        importance = model_feature_importance(models.xgb_multi, data.feature_cols)
        fig_path = plot_shap_global(importance, config)
        empty = {"png": "", "pdf": ""}
        return importance, fig_path, empty, empty, {}

    try:
        import shap

        sample_size = min(config.shap_sample_size, len(data.X_val_scaled))
        rng = np.random.default_rng(config.seed)
        sample_index = rng.choice(len(data.X_val_scaled), sample_size, replace=False)
        X_shap = data.X_val_scaled[sample_index]
        explainer = shap.TreeExplainer(models.xgb_multi)
        shap_values_raw = explainer.shap_values(X_shap)
        shap_values = shap_values_by_class(shap_values_raw, len(data.feature_cols))
        mean_abs = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
        importance = pd.DataFrame({"Feature": data.feature_cols, "Mean_abs_SHAP": mean_abs}).sort_values(
            "Mean_abs_SHAP",
            ascending=False,
        )
        global_path = plot_shap_global(importance, config)

        class_index = min(1, len(shap_values) - 1)
        beeswarm_path = plot_shap_beeswarm(shap_values[class_index], X_shap, data.feature_cols, config)

        y_subset = data.y_val_multi.iloc[sample_index].reset_index(drop=True)
        pred_subset = pd.Series(final_predictions[sample_index]).reset_index(drop=True)
        wrong_positions = np.where(pred_subset.astype(str).to_numpy() != y_subset.astype(str).to_numpy())[0]
        waterfall_pos = int(wrong_positions[0]) if len(wrong_positions) else 0
        waterfall_path = plot_shap_waterfall(
            shap,
            shap_values[class_index],
            explainer,
            X_shap,
            data.feature_cols,
            class_index,
            waterfall_pos,
            config,
        )

        top_features: dict[str, list[str]] = {}
        for index, class_name in enumerate(models.multi_encoder.classes_):
            class_values = shap_values[min(index, len(shap_values) - 1)]
            top3 = np.argsort(np.abs(class_values).mean(axis=0))[-3:][::-1]
            top_features[str(class_name)] = [data.feature_cols[position] for position in top3]
        return importance, global_path, beeswarm_path, waterfall_path, top_features
    except Exception as exc:
        warning_path.write_text(f"TreeExplainer fallback used: {exc}", encoding="utf-8")
        return generate_native_xgb_shap_outputs(data, models, final_predictions, config)


def native_xgb_shap_values(model: Any, X_shap: np.ndarray, feature_cols: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xgb = import_xgboost()
    dmatrix = xgb.DMatrix(X_shap, feature_names=feature_cols)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    values = np.asarray(contributions)
    feature_count = len(feature_cols)

    if values.ndim == 2 and values.shape[1] == feature_count + 1:
        return [values[:, :-1]], [values[:, -1]]
    if values.ndim == 2 and values.shape[1] % (feature_count + 1) == 0:
        class_count = values.shape[1] // (feature_count + 1)
        values = values.reshape(values.shape[0], class_count, feature_count + 1)
    if values.ndim == 3 and values.shape[2] == feature_count + 1:
        return [values[:, index, :-1] for index in range(values.shape[1])], [
            values[:, index, -1] for index in range(values.shape[1])
        ]
    raise ValueError(f"Unsupported native XGBoost SHAP contribution shape: {values.shape}")


def generate_native_xgb_shap_outputs(
    data: PreparedData,
    models: TrainedModels,
    final_predictions: np.ndarray,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str], dict[str, list[str]]]:
    sample_size = min(config.shap_sample_size, len(data.X_val_scaled))
    rng = np.random.default_rng(config.seed)
    sample_index = rng.choice(len(data.X_val_scaled), sample_size, replace=False)
    X_shap = data.X_val_scaled[sample_index]
    shap_values, base_values = native_xgb_shap_values(models.xgb_multi, X_shap, data.feature_cols)

    mean_abs = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
    importance = pd.DataFrame({"Feature": data.feature_cols, "Mean_abs_SHAP": mean_abs}).sort_values(
        "Mean_abs_SHAP",
        ascending=False,
    )
    global_path = plot_shap_global(importance, config)

    class_index = min(1, len(shap_values) - 1)
    beeswarm_path = plot_native_shap_beeswarm(shap_values[class_index], X_shap, data.feature_cols, config)

    y_subset = data.y_val_multi.iloc[sample_index].reset_index(drop=True)
    pred_subset = pd.Series(final_predictions[sample_index]).reset_index(drop=True)
    wrong_positions = np.where(pred_subset.astype(str).to_numpy() != y_subset.astype(str).to_numpy())[0]
    waterfall_pos = int(wrong_positions[0]) if len(wrong_positions) else 0
    waterfall_path = plot_native_shap_waterfall(
        shap_values[class_index],
        base_values[class_index],
        X_shap,
        data.feature_cols,
        waterfall_pos,
        config,
    )

    top_features: dict[str, list[str]] = {}
    for index, class_name in enumerate(models.multi_encoder.classes_):
        class_values = shap_values[min(index, len(shap_values) - 1)]
        top3 = np.argsort(np.abs(class_values).mean(axis=0))[-3:][::-1]
        top_features[str(class_name)] = [data.feature_cols[position] for position in top3]
    return importance, global_path, beeswarm_path, waterfall_path, top_features


def plot_native_shap_beeswarm(
    values: np.ndarray,
    X_shap: np.ndarray,
    feature_cols: list[str],
    config: ExperimentConfig,
) -> dict[str, str]:
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-20:]
    ordered_indices = top_indices[np.argsort(mean_abs[top_indices])]

    fig, ax = plt.subplots(figsize=(7.2, max(4.8, len(ordered_indices) * 0.30)))
    rng = np.random.default_rng(config.seed)
    for y_pos, feature_index in enumerate(ordered_indices):
        feature_values = X_shap[:, feature_index]
        denominator = np.ptp(feature_values)
        colors = (feature_values - feature_values.min()) / denominator if denominator else np.zeros_like(feature_values)
        jitter = rng.normal(0, 0.055, size=len(feature_values))
        ax.scatter(values[:, feature_index], np.full(len(feature_values), y_pos) + jitter, c=colors, cmap="coolwarm", s=12, alpha=0.72)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(range(len(ordered_indices)))
    ax.set_yticklabels([feature_cols[index] for index in ordered_indices])
    ax.set_xlabel("Native XGBoost SHAP contribution")
    ax.set_title("SHAP Beeswarm")
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, "fig_shap_beeswarm", config)


def plot_native_shap_waterfall(
    values: np.ndarray,
    base_values: np.ndarray,
    X_shap: np.ndarray,
    feature_cols: list[str],
    row_index: int,
    config: ExperimentConfig,
) -> dict[str, str]:
    row_values = values[row_index]
    top_indices = np.argsort(np.abs(row_values))[-14:]
    ordered_indices = top_indices[np.argsort(row_values[top_indices])]
    colors = ["#dc2626" if row_values[index] > 0 else "#2563eb" for index in ordered_indices]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh([feature_cols[index] for index in ordered_indices], [row_values[index] for index in ordered_indices], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    model_output = float(base_values[row_index] + row_values.sum())
    ax.set_title(f"SHAP Waterfall Sample: base={base_values[row_index]:.3f}, output={model_output:.3f}")
    ax.set_xlabel("Native XGBoost SHAP contribution")
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, "fig_shap_waterfall", config)


def plot_shap_global(importance: pd.DataFrame, config: ExperimentConfig) -> dict[str, str]:
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(4.8, len(top) * 0.28)))
    ax.barh(top["Feature"], top["Mean_abs_SHAP"], color="#2563eb")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, "fig_03_shap_global", config)


def plot_shap_beeswarm(
    values: np.ndarray,
    X_shap: np.ndarray,
    feature_cols: list[str],
    config: ExperimentConfig,
) -> dict[str, str]:
    import shap

    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(values, X_shap, feature_names=feature_cols, show=False, max_display=20)
    fig = plt.gcf()
    return save_figure(fig, "fig_shap_beeswarm", config)


def plot_shap_waterfall(
    shap: Any,
    values: np.ndarray,
    explainer: Any,
    X_shap: np.ndarray,
    feature_cols: list[str],
    class_index: int,
    row_index: int,
    config: ExperimentConfig,
) -> dict[str, str]:
    explanation = shap.Explanation(
        values=values[row_index],
        base_values=expected_value_for_class(explainer.expected_value, class_index),
        data=X_shap[row_index],
        feature_names=feature_cols,
    )
    shap.plots.waterfall(explanation, show=False, max_display=14)
    fig = plt.gcf()
    return save_figure(fig, "fig_shap_waterfall", config)


def plot_severity_distribution(severity: pd.Series, config: ExperimentConfig) -> dict[str, str]:
    counts = severity.value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    colors = ["#22c55e", "#eab308", "#f97316", "#dc2626"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("Adaptive Alert Severity Distribution")
    return save_figure(fig, "fig_04_severity_distribution", config)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, config: ExperimentConfig) -> dict[str, str]:
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.plot(sensitivity["Threshold"], sensitivity["F1_macro"], marker="o", color="#2563eb", label="Macro-F1")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Macro-F1", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(sensitivity["Threshold"], sensitivity["Unknown_pct"], marker="s", color="#dc2626", label="Unknown %")
    ax2.set_ylabel("Unknown predictions (%)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax1.set_title("Threshold Sensitivity Analysis")
    return save_figure(fig, "fig_05_threshold_sensitivity", config)


def plot_table_figure(frame: pd.DataFrame, name: str, title: str, config: ExperimentConfig) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(max(7, len(frame.columns) * 1.15), max(2.4, len(frame) * 0.55 + 1.2)))
    ax.axis("off")
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title(title, pad=16)
    return save_figure(fig, name, config)


def plot_latency_scatter(latency: pd.DataFrame, config: ExperimentConfig) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sns.scatterplot(data=latency, x="Latency_ms_per_sample", y="F1_macro", size="Size_kb", hue="Model", sizes=(80, 280), ax=ax)
    for _, row in latency.iterrows():
        ax.annotate(row["Model"], (row["Latency_ms_per_sample"], row["F1_macro"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Latency (ms/sample)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Latency vs F1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return save_figure(fig, "fig_07_latency_vs_f1", config)


def plot_severity_heatmap(y_true: pd.Series, severity: pd.Series, config: ExperimentConfig) -> dict[str, str]:
    frame = pd.DataFrame({"Attack": y_true.astype(str), "Severity": severity.astype(str)})
    top_attacks = frame["Attack"].value_counts().head(14).index
    frame = frame[frame["Attack"].isin(top_attacks)]
    heat = pd.crosstab(frame["Attack"], frame["Severity"]).reindex(columns=["Low", "Medium", "High", "Critical"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, len(heat) * 0.36)))
    sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt="d", ax=ax)
    ax.set_title("Severity vs Actual Attack Label")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Attack class")
    return save_figure(fig, "fig_08_severity_attack_heatmap", config)


def latency_table(
    data: PreparedData,
    models: TrainedModels,
    val_metrics: dict[str, float],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    samples = data.X_val_scaled[:1]
    for key, model in (("xgb_multi", models.xgb_multi), ("xgb_binary", models.xgb_binary), ("iso", models.iso)):
        start = time.perf_counter()
        for _ in range(100):
            model.predict(samples)
        latency_ms = (time.perf_counter() - start) / 100 * 1000
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            joblib.dump(model, temp_path)
            size_kb = temp_path.stat().st_size / 1024
        finally:
            if temp_path.exists():
                temp_path.unlink()
        rows.append(
            {
                "Model": MODEL_NAMES[key],
                "Latency_ms_per_sample": float(latency_ms),
                "Size_kb": float(size_kb),
                "F1_macro": float(val_metrics["F1_macro"]),
            }
        )
    return pd.DataFrame(rows)


def ablation_study(
    data: PreparedData,
    models: TrainedModels,
    xgb_val_pred: np.ndarray,
    xgb_val_conf: np.ndarray,
    xgb_val_proba: np.ndarray,
    final_pred: np.ndarray,
    severity: pd.Series,
    config: ExperimentConfig,
) -> pd.DataFrame:
    baseline_model, baseline_encoder = train_xgb_classifier(
        data.X_train_scaled,
        data.y_train_multi,
        data.X_val_scaled,
        data.y_val_multi,
        config,
    )
    base_pred, _, base_proba = predict_encoded(baseline_model, baseline_encoder, data.X_val_scaled)
    base_metrics = multiclass_metrics(data.y_val_multi, base_pred, base_proba, baseline_encoder)

    smote_metrics = multiclass_metrics(data.y_val_multi, xgb_val_pred, xgb_val_proba, models.multi_encoder)

    iso_flagged = models.iso.predict(data.X_val_scaled) == -1
    hybrid_pred = np.asarray(xgb_val_pred, dtype=object)
    hybrid_pred[iso_flagged] = UNKNOWN_LABEL
    hybrid_metrics = multiclass_metrics(data.y_val_multi, hybrid_pred, xgb_val_proba, models.multi_encoder)

    full_metrics = multiclass_metrics(data.y_val_multi, final_pred, xgb_val_proba, models.multi_encoder)
    target_severity = target_severity_from_attack(data.y_val_multi)
    severity_acc = float((severity.reset_index(drop=True) == target_severity).mean())

    rows = [
        {
            "Config": "A) XGBoost only, no SMOTE, no threshold",
            "F1_macro": base_metrics["F1_macro"],
            "MCC": base_metrics["MCC"],
            "Unknown_pct": np.nan,
            "Severity_acc": np.nan,
        },
        {
            "Config": "B) XGBoost + SMOTE, no threshold",
            "F1_macro": smote_metrics["F1_macro"],
            "MCC": smote_metrics["MCC"],
            "Unknown_pct": np.nan,
            "Severity_acc": np.nan,
        },
        {
            "Config": "C) XGBoost + Isolation Forest, no confidence threshold",
            "F1_macro": hybrid_metrics["F1_macro"],
            "MCC": hybrid_metrics["MCC"],
            "Unknown_pct": float(iso_flagged.mean() * 100),
            "Severity_acc": severity_acc,
        },
        {
            "Config": "D) Full hybrid + unknown threshold + severity",
            "F1_macro": full_metrics["F1_macro"],
            "MCC": full_metrics["MCC"],
            "Unknown_pct": float((final_pred == UNKNOWN_LABEL).mean() * 100),
            "Severity_acc": severity_acc,
        },
    ]
    return pd.DataFrame(rows)


def generalization_gap_table(
    internal_metrics: dict[str, float],
    external_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for metric in ("Accuracy", "F1_macro", "AUC_ROC"):
        internal = internal_metrics.get(metric, float("nan"))
        external = external_metrics.get(metric, float("nan"))
        rows.append(
            {
                "Metric": metric,
                "NF-ToN-IoT-V2": internal,
                "NF-UNSW-NB15": external,
                "Drop": internal - external if pd.notna(internal) and pd.notna(external) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def final_metrics_table(
    val_metrics: dict[str, float],
    internal_binary_metrics: dict[str, float],
    external_metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Dataset": "NF-ToN-IoT-V2 validation multiclass", **val_metrics},
            {"Dataset": "NF-ToN-IoT-V2 validation binary", **internal_binary_metrics},
            {"Dataset": "NF-UNSW-NB15 external binary", **external_metrics},
        ]
    )


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    setup_seed(config.seed)
    ensure_dirs(config)

    log_step(0, "Global seed configured", {"seed": config.seed, "pythonhashseed": os.environ.get("PYTHONHASHSEED")})

    data = prepare_data(config)
    log_step(
        1,
        "Data pipeline complete",
        {
            "train_data": config.train_data,
            "features": len(data.feature_cols),
            "train_rows": len(data.y_train_multi),
            "val_rows": len(data.y_val_multi),
            "class_dist_before": data.class_distribution_before,
            "class_dist_after": data.class_distribution_after,
            "smote": data.smote_note,
            "binary_class_dist_after": data.binary_distribution_after,
            "binary_smote": data.binary_smote_note,
        },
    )

    models = train_models(data, config)
    xgb_val_pred, xgb_val_conf, xgb_val_proba = predict_encoded(models.xgb_multi, models.multi_encoder, data.X_val_scaled)
    raw_val_score = models.iso.decision_function(data.X_val_scaled)
    ano_score, anomaly_scaler = normalize_anomaly_scores(raw_val_score)
    model_paths = save_models(data, models, config)
    log_step(
        2,
        "Hybrid model trained and saved",
        {
            "xgb_classes": list(models.multi_encoder.classes_),
            "model_dir": config.model_dir,
            "mean_confidence": round(float(np.mean(xgb_val_conf)), 6),
            "mean_anomaly_score": round(float(np.mean(ano_score)), 6),
        },
    )

    final_pred = apply_unknown_threshold(xgb_val_pred, xgb_val_conf, config.confidence_threshold)
    sensitivity = threshold_sensitivity(data.y_val_multi, xgb_val_pred, xgb_val_conf)
    log_step(
        3,
        "Unknown attack detection complete",
        {
            "confidence_threshold": config.confidence_threshold,
            "unknown_pct": round(float((final_pred == UNKNOWN_LABEL).mean() * 100), 4),
            "sensitivity_rows": len(sensitivity),
        },
    )

    severity, severity_score = get_severity(xgb_val_conf, ano_score, config.alpha)
    severity_summary = severity.value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0).astype(int).to_dict()
    severity_by_attack = pd.DataFrame({"Attack": data.y_val_multi, "Severity_score": severity_score}).groupby("Attack", as_index=False)[
        "Severity_score"
    ].mean()
    log_step(
        4,
        "Adaptive severity scoring complete",
        {
            "alpha": config.alpha,
            "severity_distribution": severity_summary,
            "mean_severity_score": round(float(np.mean(severity_score)), 6),
        },
    )

    shap_importance, shap_global_path, shap_beeswarm_path, shap_waterfall_path, top_features = generate_shap_outputs(
        data,
        models,
        final_pred,
        config,
    )
    log_step(
        5,
        "SHAP explainability complete",
        {
            "top_global_features": shap_importance.head(3)["Feature"].tolist(),
            "shap_global_pdf": shap_global_path.get("pdf", ""),
            "beeswarm_pdf": shap_beeswarm_path.get("pdf", ""),
            "waterfall_pdf": shap_waterfall_path.get("pdf", ""),
        },
    )

    external_table, external_summary = evaluate_external_dataset(data, models, anomaly_scaler, config)
    internal_binary_pred = predictions_to_binary(final_pred)
    internal_binary_proba = infer_positive_probability(models.xgb_binary, models.binary_encoder, models.xgb_binary.predict_proba(data.X_val_scaled))
    internal_binary_metrics = binary_metrics(data.y_val_bin, internal_binary_pred, internal_binary_proba)
    gap = generalization_gap_table(internal_binary_metrics, external_summary["metrics"])
    log_step(
        6,
        "Cross-dataset validation complete",
        {
            "external_data": config.external_data,
            "external_rows": external_summary["rows"],
            "external_accuracy": round(external_summary["metrics"]["Accuracy"], 6),
            "external_auc": external_summary["metrics"].get("AUC_ROC"),
        },
    )

    val_metrics = multiclass_metrics(data.y_val_multi, final_pred, xgb_val_proba, models.multi_encoder)
    f1_frame = per_class_f1_frame(data.y_val_multi, final_pred)
    latency = latency_table(data, models, val_metrics, config)
    metrics = final_metrics_table(val_metrics, internal_binary_metrics, external_summary["metrics"])
    log_step(
        7,
        "Final metrics and paper figures prepared",
        {
            "validation_accuracy": round(val_metrics["Accuracy"], 6),
            "validation_f1_macro": round(val_metrics["F1_macro"], 6),
            "latency_rows": len(latency),
        },
    )

    ablation = ablation_study(
        data,
        models,
        xgb_val_pred,
        xgb_val_conf,
        xgb_val_proba,
        final_pred,
        severity,
        config,
    )
    log_step(
        8,
        "Ablation study complete",
        {
            "configs": len(ablation),
            "proposed_f1": round(float(ablation.iloc[-1]["F1_macro"]), 6),
            "proposed_unknown_pct": round(float(ablation.iloc[-1]["Unknown_pct"]), 4),
        },
    )

    table_paths = {
        "table_i_final_metrics": save_table(metrics, "table_i_final_metrics", config),
        "table_ii_ablation": save_table(ablation, "table_ii_ablation", config),
        "table_iii_threshold_sensitivity": save_table(sensitivity, "table_iii_threshold_sensitivity", config),
        "table_iv_generalization_gap": save_table(gap, "table_iv_generalization_gap", config),
        "severity_by_attack": save_table(severity_by_attack, "severity_by_attack", config),
        "shap_top_features": save_table(
            pd.DataFrame(
                [{"Attack": attack, "Top_3_features": ", ".join(features)} for attack, features in top_features.items()]
            ),
            "shap_top_features",
            config,
        ),
        "latency": save_table(latency, "latency", config),
        "external_metrics": save_table(external_table, "external_metrics", config),
    }

    figure_paths = {
        "fig_01_confusion_matrix": plot_confusion_matrix(data.y_val_multi, final_pred, config),
        "fig_02_per_class_f1": plot_per_class_f1(f1_frame, config),
        "fig_03_shap_global": shap_global_path,
        "fig_04_severity_distribution": plot_severity_distribution(severity, config),
        "fig_05_threshold_sensitivity": plot_threshold_sensitivity(sensitivity, config),
        "fig_06_cross_dataset_generalization": plot_table_figure(
            gap,
            "fig_06_cross_dataset_generalization",
            "Cross-Dataset Generalization",
            config,
        ),
        "fig_07_latency_vs_f1": plot_latency_scatter(latency, config),
        "fig_08_severity_attack_heatmap": plot_severity_heatmap(data.y_val_multi, severity, config),
        "fig_shap_beeswarm": shap_beeswarm_path,
        "fig_shap_waterfall": shap_waterfall_path,
    }

    manifest = {
        "claim": "Lightweight hybrid uncertainty-aware NIDS with cross-dataset generalization and explainable intrusion analysis",
        "config": jsonable(asdict(config)),
        "feature_columns": data.feature_cols,
        "class_distribution_before_smote": data.class_distribution_before,
        "class_distribution_after_smote": data.class_distribution_after,
        "binary_distribution_after_smote": data.binary_distribution_after,
        "validation_metrics": val_metrics,
        "internal_binary_metrics": internal_binary_metrics,
        "external_summary": external_summary,
        "severity_distribution": severity_summary,
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "generalization_gap": gap.to_dict(orient="records"),
        "top_features_per_attack": top_features,
        "models": model_paths,
        "tables": table_paths,
        "figures": figure_paths,
    }
    manifest_path = config.output_dir / "journal_upgrade_manifest.json"
    manifest_path.write_text(json.dumps(jsonable(manifest), indent=2), encoding="utf-8")
    print(f"[DONE] Journal upgrade artifacts written to {config.output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run journal-ready XGBoost + Isolation Forest NIDS experiment.")
    parser.add_argument("--train-data", type=Path, default=default_train_data())
    parser.add_argument("--external-data", type=Path, default=default_external_data())
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "journal_upgrade")
    parser.add_argument("--figure-dir", type=Path, default=PAPER_DIR / "figures" / "journal_upgrade")
    parser.add_argument("--table-dir", type=Path, default=PAPER_DIR / "generated" / "journal_upgrade")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR / "journal_upgrade")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-external-rows", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=CONF_THR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--shap-sample-size", type=int, default=500)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--iso-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast artifact smoke test with small row limits and lighter models.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        args.max_train_rows = args.max_train_rows or 5000
        args.max_external_rows = args.max_external_rows or 5000
        args.xgb_estimators = min(args.xgb_estimators, 40)
        args.iso_estimators = min(args.iso_estimators, 30)
        args.shap_sample_size = min(args.shap_sample_size, 80)

    return ExperimentConfig(
        train_data=args.train_data,
        external_data=args.external_data,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        table_dir=args.table_dir,
        model_dir=args.model_dir,
        max_train_rows=args.max_train_rows,
        max_external_rows=args.max_external_rows,
        sample_mode=args.sample_mode,
        test_size=args.test_size,
        confidence_threshold=args.confidence_threshold,
        alpha=args.alpha,
        shap_sample_size=args.shap_sample_size,
        skip_shap=args.skip_shap,
        xgb_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        iso_estimators=args.iso_estimators,
        n_jobs=args.n_jobs,
        seed=args.seed,
        use_smote=not args.no_smote,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
