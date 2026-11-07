"""Compare Task 1, Task 2, and Task 3 test predictions."""

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
    parser.add_argument("--output", default="checkpoints/task3/comparison.json")
    args = parser.parse_args()
    task1 = pd.read_csv(args.task1_preds)
    task2 = pd.read_csv(args.task2_preds)
    task3 = pd.read_csv(args.task3_preds)
    merged = task1[["sentence_id", "gold_label", "task1_prediction"]].merge(
        task2[["sentence_id", "task2_prediction"]], on="sentence_id", validate="one_to_one"
    ).merge(task3[["sentence_id", "task3_prediction"]], on="sentence_id", validate="one_to_one")
    result = {
        "task1": metrics(merged["gold_label"].tolist(), merged["task1_prediction"].tolist()),
        "task2": metrics(merged["gold_label"].tolist(), merged["task2_prediction"].tolist()),
        "task3": metrics(merged["gold_label"].tolist(), merged["task3_prediction"].tolist()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
