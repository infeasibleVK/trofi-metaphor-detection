"""Evaluate a trained Task 2 checkpoint on a TroFi split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import TroFiDataset, make_collate_fn
from embeddings import ElmoEncoder, load_glove
from model import TroFiBiLSTM
from train import run_epoch


def predict(model, elmo, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            embeddings = elmo(batch.tokens, device)
            logits = model(batch.token_ids, embeddings, batch.target_indices)
            predictions = logits.argmax(dim=1).cpu().tolist()
            for example_id, label, prediction in zip(
                batch.example_ids, batch.labels.cpu().tolist(), predictions
            ):
                rows.append(
                    {
                        "sentence_id": example_id,
                        "gold_label": label,
                        "task2_prediction": prediction,
                        "is_correct": int(label == prediction),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--elmo_options", required=True)
    parser.add_argument("--elmo_weights", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)
    vocabulary, glove_weights = load_glove(args.glove_path)
    if vocabulary != checkpoint["vocabulary"]:
        raise ValueError("The evaluation GloVe vocabulary differs from the training checkpoint")
    dataset = TroFiDataset(args.test_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(vocabulary),
    )
    elmo = ElmoEncoder(args.elmo_options, args.elmo_weights, device)
    model = TroFiBiLSTM(glove_weights, elmo_dim=elmo.output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = predict(model, elmo, loader, device)
    labels = [row["gold_label"] for row in rows]
    predictions = [row["task2_prediction"] for row in rows]

    result = {
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["literal", "metaphorical"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    print(json.dumps(result, indent=2))
    output_path = Path(args.model_path).with_name("test_metrics.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    prediction_path = Path(args.model_path).with_name("test_predictions.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(prediction_path, index=False)


if __name__ == "__main__":
    main()
