"""Compute point-biserial correlation between SPV and neural correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spv_scores", default="data/task1/spv_scores_test.csv")
    parser.add_argument("--neural_preds", default="checkpoints/task2/test_predictions.csv")
    parser.add_argument("--output", default="checkpoints/task3/correlation.json")
    args = parser.parse_args()

    scores = load_features(args.spv_scores)
    predictions = pd.read_csv(args.neural_preds)
    required = {"sentence_id", "is_correct"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{args.neural_preds} is missing columns: {sorted(missing)}")
    merged = scores.merge(
        predictions[["sentence_id", "is_correct"]],
        on="sentence_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(scores):
        raise ValueError("SPV and neural prediction IDs do not match exactly")
    x = merged["association_score"].astype(float).to_numpy()
    y = merged["is_correct"].astype(float).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("association_score contains non-finite values")
    if np.std(x) == 0 or np.std(y) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    result = {
        "measure": "point_biserial_correlation",
        "score_column": "association_score",
        "correctness_column": "is_correct",
        "n": len(merged),
        "correlation": correlation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
