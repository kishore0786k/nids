from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ns_nids_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "NF-UNSW-NB15-v2.csv"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "cross_dataset_results.json"
NIDS_DATASET_PAGE = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")

    def handle_data(self, data: str) -> None:
        if self._href:
            self.links.append((data.strip(), self._href))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._href = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained NF-ToN-IoT-V2 model on NF-UNSW-NB15.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--reference_path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--download_url", default=NIDS_DATASET_PAGE)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def discover_nf_unsw_v2_url(page_url: str) -> str:
    with urllib.request.urlopen(page_url, timeout=60) as response:
        html = response.read().decode("utf-8", errors="ignore")

    match = re.search(r"NF-UNSW-NB15-v2.*?href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return urllib.request.urljoin(page_url, match.group(1))

    parser = LinkParser()
    parser.feed(html)
    for _, href in parser.links:
        if "rdm.uq.edu.au" in href:
            return urllib.request.urljoin(page_url, href)
    raise RuntimeError(f"Could not discover NF-UNSW-NB15-v2 download link from {page_url}")


def download_nf_unsw(data_path: Path, source_url: str) -> Path:
    if data_path.exists():
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_url = source_url if "rdm.uq.edu.au" in source_url else discover_nf_unsw_v2_url(source_url)
    temporary_path = data_path.with_suffix(data_path.suffix + ".download")
    print(f"Downloading NF-UNSW-NB15 from {resolved_url}")
    urllib.request.urlretrieve(resolved_url, temporary_path)

    if zipfile.is_zipfile(temporary_path):
        with zipfile.ZipFile(temporary_path) as archive:
            csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
            if not csv_members:
                raise RuntimeError(f"Downloaded archive has no CSV files: {temporary_path}")
            member = next((name for name in csv_members if "unsw" in name.lower()), csv_members[0])
            with archive.open(member) as source, data_path.open("wb") as target:
                target.write(source.read())
        temporary_path.unlink(missing_ok=True)
    else:
        temporary_path.replace(data_path)
    return data_path


def feature_columns(model: Any, reference_path: Path) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    reference = pd.read_csv(reference_path, nrows=1)
    label_col = "label" if "label" in reference.columns else "Label"
    return [column for column in reference.columns if column != label_col]


def align_features(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    aligned = frame.copy()
    missing = [column for column in columns if column not in aligned.columns]
    extra = [column for column in aligned.columns if column not in columns and column not in {"Label", "Attack", "label"}]
    for column in missing:
        aligned[column] = 0
    X = aligned[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, missing, extra


def map_attack_label(raw_label: Any, model_classes: list[str]) -> str:
    label = str(raw_label).strip()
    lower = label.lower()
    class_set = set(model_classes)
    if label in class_set:
        return label
    if lower in {"benign", "normal"}:
        return "Benign" if "Benign" in class_set else label
    if lower in {"dos", "ddos"}:
        return "DoS/DDoS" if "DoS/DDoS" in class_set else label
    if lower in {"reconnaissance", "recon", "scanning"}:
        return "Scanning" if "Scanning" in class_set else label
    if lower in {"backdoor"}:
        return "Backdoor" if "Backdoor" in class_set else label
    if lower in {"analysis", "exploits", "fuzzers", "generic", "shellcode", "worms"}:
        return "UNKNOWN"
    return label


def load_labels(frame: pd.DataFrame, model_classes: list[str]) -> pd.Series:
    if "Attack" in frame.columns:
        return frame["Attack"].map(lambda value: map_attack_label(value, model_classes)).astype(str)
    if "Label" in frame.columns:
        return frame["Label"].map(lambda value: "Benign" if int(value) == 0 else "UNKNOWN").astype(str)
    if "label" in frame.columns:
        return frame["label"].astype(str)
    raise ValueError("External dataset must contain Attack, Label, or label.")


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = download_nf_unsw(args.data_path, args.download_url)

    model = joblib.load(args.model_path)
    columns = feature_columns(model, args.reference_path)
    model_classes = [str(label) for label in getattr(model, "classes_", [])]

    frame = pd.read_csv(data_path, nrows=args.max_rows)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[column for column in ["Attack", "Label", "label"] if column in frame.columns])
    X, missing, extra = align_features(frame, columns)
    y_true = load_labels(frame, model_classes)

    model_input = X if getattr(model, "feature_names_in_", None) is not None else X.to_numpy()
    y_pred = pd.Series([str(label) for label in model.predict(model_input)])
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)) | set(model_classes))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_f1 = {
        label: float(report.get(label, {}).get("f1-score", 0.0))
        for label in labels
    }

    results = {
        "model_path": args.model_path,
        "data_path": data_path,
        "download_source": args.download_url,
        "rows": int(len(X)),
        "feature_alignment": {
            "expected_features": len(columns),
            "missing_filled_with_zero": missing,
            "extra_columns_dropped": extra,
        },
        "labels": labels,
        "per_class_f1": per_class_f1,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
    }
    args.output_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(json.dumps(json_safe({"output_path": args.output_path, "macro_f1": results["macro_f1"], "rows": len(X)}), indent=2))


if __name__ == "__main__":
    main()
