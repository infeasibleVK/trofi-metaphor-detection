"""Analyze Task 1 and Task 2 disagreements and Task 3 outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_preds", default="data/task1_preds.csv")
    parser.add_argument("--task2_preds", default="checkpoints/task2/test_predictions.csv")
    parser.add_argument("--task3_preds", default="checkpoints/task3/test_predictions.csv")
    parser.add_argument("--output", default="checkpoints/task3/disagreement.json")
    args = parser.parse_args()

    task1 = pd.read_csv(args.task1_preds)
    task2 = pd.read_csv(args.task2_preds)
    task3 = pd.read_csv(args.task3_preds)
    required = {"sentence_id", "gold_label"}
    for name, frame in (("Task 1", task1), ("Task 2", task2), ("Task 3", task3)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} predictions need columns {sorted(required)}")
    task1 = task1[["sentence_id", "gold_label", "task1_prediction"]]
    task2 = task2[["sentence_id", "task2_prediction"]]
    task3 = task3[["sentence_id", "task3_prediction"]]
    merged = task1.merge(task2, on="sentence_id", validate="one_to_one").merge(
        task3, on="sentence_id", validate="one_to_one"
    )
    if len(merged) != len(task1):
        raise ValueError("Prediction files do not contain the same sentence IDs")
    disagreement = merged[merged["task1_prediction"] != merged["task2_prediction"]].copy()
    disagreement["task1_correct"] = (
        disagreement["task1_prediction"] == disagreement["gold_label"]
    )
    groups = {}
    for name, group in disagreement.groupby("task1_correct", dropna=False):
        key = "task1_correct_task2_incorrect" if name else "task1_incorrect_task2_correct"
        groups[key] = {
            "n": len(group),
            **metrics(
                group["gold_label"].tolist(), group["task3_prediction"].tolist()
            ),
        }
    result = {
        "total_test_examples": len(merged),
        "disagreement_examples": len(disagreement),
        "disagreement_rate": len(disagreement) / len(merged) if len(merged) else 0.0,
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
