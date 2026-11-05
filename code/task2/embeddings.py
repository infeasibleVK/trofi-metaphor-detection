"""GloVe loading and optional AllenNLP ELMo feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch


def load_glove(path: str | Path) -> tuple[dict[str, int], torch.Tensor]:
    """Load a text GloVe file and prepend padding/unknown rows."""
    vectors: list[list[float]] = []
    vocabulary = {"<pad>": 0, "<unk>": 1}
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.rstrip().split(" ")
            if len(fields) < 3:
                raise ValueError(f"Malformed GloVe line {line_number}")
            word, values = fields[0], fields[1:]
            try:
                vector = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(f"Invalid vector on GloVe line {line_number}") from exc
            if word in vocabulary:
                continue
            vocabulary[word] = len(vectors) + 2
            vectors.append(vector)
    if not vectors:
        raise ValueError(f"No vectors found in {path}")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("GloVe vectors have inconsistent dimensions")
    matrix = torch.zeros((len(vectors) + 2, dimension), dtype=torch.float32)
    matrix[1] = torch.tensor(vectors, dtype=torch.float32).mean(dim=0)
    matrix[2:] = torch.tensor(vectors, dtype=torch.float32)
    return vocabulary, matrix


class ElmoEncoder:
    """Thin wrapper that keeps the optional ELMo dependency out of imports."""

    def __init__(self, options_file: str, weight_file: str, device: torch.device) -> None:
        try:
            from allennlp.modules.elmo import Elmo, batch_to_ids
        except ImportError as exc:
            raise ImportError(
                "ELMo requires allennlp. Install the pinned project dependencies "
                "before training Task 2."
            ) from exc
        self._batch_to_ids = batch_to_ids
        self._model = Elmo(
            options_file,
            weight_file,
            num_output_representations=1,
            dropout=0.0,
            requires_grad=False,
        ).to(device)
        self._model.eval()

    @property
    def output_dim(self) -> int:
        return 1024

    @torch.no_grad()
    def __call__(self, tokens: list[list[str]], device: torch.device) -> torch.Tensor:
        character_ids = self._batch_to_ids(tokens).to(device)
        return self._model(character_ids)["elmo_representations"][0]
