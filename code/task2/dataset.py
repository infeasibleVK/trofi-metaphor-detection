"""Dataset and batching utilities for TroFi target-verb classification."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset


def _parse_tokens(value: Any, sentence: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return sentence.split()
    if isinstance(value, list):
        return [str(token) for token in value]
    text = str(value).strip()
    if not text:
        return sentence.split()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [str(token) for token in parsed]
    return text.split()


def _first_value(row: pd.Series, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name not in row.index:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, str) or not hasattr(value, "__len__"):
            if not pd.isna(value):
                return value
        else:
            return value
    return default


def _target_index(
    tokens: list[str], verb: str, supplied: Any, target_token: Any = None
) -> int:
    if supplied is not None and not pd.isna(supplied):
        index = int(supplied)
        expected = {str(verb).lower()}
        if target_token is not None and not pd.isna(target_token):
            expected.add(str(target_token).lower())
        if 0 <= index < len(tokens) and tokens[index].lower() in expected:
            return index

    verb = str(verb).lower()
    target_token = (
        str(target_token).lower()
        if target_token is not None and not pd.isna(target_token)
        else ""
    )
    candidates = [
        i for i, token in enumerate(tokens)
        if token.lower() == verb or (target_token and token.lower() == target_token)
    ]
    if not candidates:
        candidates = [
            i for i, token in enumerate(tokens)
            if token.lower().startswith(verb) and len(token) > len(verb)
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not uniquely locate target verb {verb!r} in: {' '.join(tokens)}"
        )
    return candidates[0]


@dataclass(frozen=True)
class TroFiExample:
    example_id: str
    tokens: list[str]
    target_index: int
    label: int
    verb: str


class TroFiDataset(Dataset[TroFiExample]):
    """Loads a Task 1-generated TroFi CSV without changing its source data."""

    def __init__(self, path: str | Path) -> None:
        frame = pd.read_csv(path)
        self.examples = [
            self._row_to_example(row, row_number)
            for row_number, (_, row) in enumerate(frame.iterrows())
        ]

    @staticmethod
    def _row_to_example(row: pd.Series, row_number: int) -> TroFiExample:
        sentence = str(
            _first_value(row, ("sentence", "text", "tokenized_sentence"), "")
        )
        verb = str(_first_value(row, ("verb", "target_verb", "target"), ""))
        tokens = _parse_tokens(_first_value(row, ("tokens", "tokenized_sentence")), sentence)
        supplied_index = _first_value(
            row, ("target_index", "target_idx", "verb_idx", "verb_index")
        )
        target_token = _first_value(row, ("target_token",))
        index = _target_index(tokens, verb, supplied_index, target_token)
        raw_label = _first_value(row, ("label", "y", "target_label"))
        if raw_label is None:
            raise ValueError(f"Missing label in row {row_number}")
        if isinstance(raw_label, str):
            raw_label = {"L": 0, "N": 1, "literal": 0, "nonliteral": 1}.get(
                raw_label.strip().lower(), raw_label
            )
        label = int(raw_label)
        if label not in (0, 1):
            raise ValueError(f"Expected binary label in row {row_number}, got {label}")
        example_id = str(
            _first_value(
                row, ("id", "sentence_id", "wsj_id", "example_id"), row_number
            )
        )
        return TroFiExample(example_id, tokens, index, label, verb)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TroFiExample:
        return self.examples[index]


@dataclass
class TroFiBatch:
    tokens: list[list[str]]
    token_ids: torch.Tensor
    lengths: torch.Tensor
    target_indices: torch.Tensor
    labels: torch.Tensor
    example_ids: list[str]

    def to(self, device: torch.device) -> "TroFiBatch":
        self.token_ids = self.token_ids.to(device)
        self.lengths = self.lengths.to(device)
        self.target_indices = self.target_indices.to(device)
        self.labels = self.labels.to(device)
        return self


def make_collate_fn(vocabulary: dict[str, int], pad_token: str = "<pad>"):
    pad_id = vocabulary[pad_token]
    unk_id = vocabulary["<unk>"]

    def collate(examples: list[TroFiExample]) -> TroFiBatch:
        max_length = max(len(example.tokens) for example in examples)
        token_ids = torch.full((len(examples), max_length), pad_id, dtype=torch.long)
        for row, example in enumerate(examples):
            ids = [vocabulary.get(token.lower(), unk_id) for token in example.tokens]
            token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return TroFiBatch(
            tokens=[example.tokens for example in examples],
            token_ids=token_ids,
            lengths=torch.tensor([len(e.tokens) for e in examples], dtype=torch.long),
            target_indices=torch.tensor(
                [e.target_index for e in examples], dtype=torch.long
            ),
            labels=torch.tensor([e.label for e in examples], dtype=torch.long),
            example_ids=[e.example_id for e in examples],
        )

    return collate
