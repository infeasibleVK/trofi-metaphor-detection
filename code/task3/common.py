"""Shared Task 3 data and metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK2_DIR = Path(__file__).resolve().parents[1] / "task2"
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from dataset import TroFiDataset, TroFiBatch, make_collate_fn  # noqa: E402


FEATURE_COLUMNS = ("association_score", "cold_start_flag")


def load_features(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sentence_id", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing Task 1 feature columns: {sorted(missing)}")
    if frame["sentence_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sentence_id values")
    return frame[["sentence_id", *FEATURE_COLUMNS]].copy()


def attach_features(dataset: TroFiDataset, feature_path: str | Path) -> pd.DataFrame:
    features = load_features(feature_path)
    ids = {example.example_id for example in dataset.examples}
    feature_ids = set(features["sentence_id"].astype(str))
    missing = ids - feature_ids
    extra = feature_ids - ids
    if missing or extra:
        raise ValueError(
            f"Task 1/Task 2 IDs do not match for {feature_path}; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    ordered = features.set_index("sentence_id").loc[
        [example.example_id for example in dataset.examples]
    ]
    return ordered.reset_index(drop=True)


def fit_feature_normalizer(features: pd.DataFrame) -> tuple[float, float]:
    scores = features["association_score"].astype(float).to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Training association_score contains non-finite values")
    mean = float(scores.mean())
    std = float(scores.std())
    return mean, std if std > 0 else 1.0


def feature_tensor(
    features: pd.DataFrame, mean: float, std: float, device
):
    scores = (features["association_score"].astype(float) - mean) / std
    cold = features["cold_start_flag"].astype(float)
    values = np.column_stack((scores.to_numpy(), cold.to_numpy()))
    if not np.isfinite(values).all():
        raise ValueError("Task 1 features contain non-finite values")
    import torch
    return torch.tensor(values, dtype=torch.float32, device=device)


def metrics(labels: list[int], predictions: list[int]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["literal", "metaphorical"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": report["metaphorical"]["precision"],
        "recall": report["metaphorical"]["recall"],
        "f1": report["metaphorical"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
