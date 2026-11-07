"""Train and evaluate the SPV-augmented BiLSTM for Task 3."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import attach_features, feature_tensor, fit_feature_normalizer, metrics
from embeddings import ElmoEncoder, load_glove
from model import TroFiBiLSTM


class SPVAugmentedBiLSTM(TroFiBiLSTM):
    def __init__(self, glove_weights, **kwargs):
        super().__init__(glove_weights, **kwargs)
        self.classifier = nn.Linear(kwargs.get("hidden_size", 300) * 2 + 2, 2)

    def forward(self, token_ids, elmo_embeddings, target_indices, features):
        glove_embeddings = self.glove(token_ids)
        encoded, _ = self.encoder(torch.cat((glove_embeddings, elmo_embeddings), dim=-1))
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        target_states = encoded[batch_indices, target_indices]
        return self.classifier(self.dropout(torch.cat((target_states, features), dim=-1)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, elmo, loader, feature_values, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, labels, rows = [], [], []
    offset = 0
    for batch in loader:
        batch.to(device)
        count = len(batch.labels)
        batch_features = feature_values[offset : offset + count]
        offset += count
        with torch.set_grad_enabled(training):
            embeddings = elmo(batch.tokens, device)
            logits = model(
                batch.token_ids, embeddings, batch.target_indices, batch_features
            )
            loss = criterion(logits, batch.labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_predictions = logits.argmax(dim=1).cpu().tolist()
        batch_labels = batch.labels.cpu().tolist()
        total_loss += loss.item() * count
        predictions.extend(batch_predictions)
        labels.extend(batch_labels)
        rows.extend(
            {
                "sentence_id": example_id,
                "gold_label": label,
                "task3_prediction": prediction,
                "is_correct": int(label == prediction),
            }
            for example_id, label, prediction in zip(
                batch.example_ids, batch_labels, batch_predictions
            )
        )
    if offset != len(feature_values):
        raise RuntimeError("Feature rows and dataset rows were not consumed equally")
    return total_loss / len(loader.dataset), predictions, labels, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/trofi_train.csv")
    parser.add_argument("--val_path", default="data/trofi_val.csv")
    parser.add_argument("--test_path", default="data/trofi_test.csv")
    parser.add_argument("--train_features", default="data/task1/spv_scores_train.csv")
    parser.add_argument("--val_features", default="data/task1/spv_scores_val.csv")
    parser.add_argument("--test_features", default="data/task1/spv_scores_test.csv")
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--elmo_options", required=True)
    parser.add_argument("--elmo_weights", required=True)
    parser.add_argument("--save_dir", default="checkpoints/task3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = __import__("dataset").TroFiDataset(args.train_path)
    val_dataset = __import__("dataset").TroFiDataset(args.val_path)
    test_dataset = __import__("dataset").TroFiDataset(args.test_path)
    train_features = attach_features(train_dataset, args.train_features)
    val_features = attach_features(val_dataset, args.val_features)
    test_features = attach_features(test_dataset, args.test_features)
    mean, std = fit_feature_normalizer(train_features)
    vocabulary, glove_weights = load_glove(args.glove_path)
    collate = __import__("dataset").make_collate_fn(vocabulary)
    loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        for ds in (train_dataset, val_dataset, test_dataset)
    ]
    feature_values = [
        feature_tensor(features, mean, std, device)
        for features in (train_features, val_features, test_features)
    ]
    elmo = ElmoEncoder(args.elmo_options, args.elmo_weights, device)
    model = SPVAugmentedBiLSTM(
        glove_weights, elmo_dim=elmo.output_dim, hidden_size=300, dropout=0.5
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model, elmo, loaders[0], feature_values[0], criterion, device, optimizer
        )
        val_loss, val_predictions, val_labels, _ = run_epoch(
            model, elmo, loaders[1], feature_values[1], criterion, device
        )
        val_metrics = metrics(val_labels, val_predictions)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{
            f"val_{key}": value for key, value in val_metrics.items()
            if key in ("accuracy", "precision", "recall", "f1", "macro_f1")
        }}
        history.append(record)
        print(json.dumps(record))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "config": vars(args),
                    "feature_mean": mean,
                    "feature_std": std,
                    "best_val_f1": best_f1,
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, predictions, labels, rows = run_epoch(
        model, elmo, loaders[2], feature_values[2], criterion, device
    )
    result = metrics(labels, predictions)
    (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
