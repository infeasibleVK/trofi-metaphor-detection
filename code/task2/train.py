"""Train the Task 2 TroFi BiLSTM model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TroFiDataset, make_collate_fn
from embeddings import ElmoEncoder, load_glove
from model import TroFiBiLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, elmo, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    for batch in loader:
        batch.to(device)
        with torch.set_grad_enabled(training):
            elmo_embeddings = elmo(batch.tokens, device)
            logits = model(batch.token_ids, elmo_embeddings, batch.target_indices)
            loss = criterion(logits, batch.labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(batch.labels)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(batch.labels.cpu().tolist())
    return total_loss / len(loader.dataset), predictions, labels


def f1_score(predictions: list[int], labels: list[int]) -> float:
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--elmo_options", required=True)
    parser.add_argument("--elmo_weights", required=True)
    parser.add_argument("--save_dir", default="checkpoints/task2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = TroFiDataset(args.train_path)
    val_dataset = TroFiDataset(args.val_path)
    vocabulary, glove_weights = load_glove(args.glove_path)
    collate = make_collate_fn(vocabulary)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    elmo = ElmoEncoder(args.elmo_options, args.elmo_weights, device)
    model = TroFiBiLSTM(glove_weights, elmo_dim=elmo.output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, elmo, train_loader, criterion, device, optimizer
        )
        val_loss, val_predictions, val_labels = run_epoch(
            model, elmo, val_loader, criterion, device
        )
        val_f1 = f1_score(val_predictions, val_labels)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        history.append(record)
        print(json.dumps(record))
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "config": vars(args),
                    "best_val_f1": best_f1,
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
