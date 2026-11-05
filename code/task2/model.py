"""GloVe + ELMo two-layer BiLSTM for TroFi target classification."""

from __future__ import annotations

import torch
from torch import nn


class TroFiBiLSTM(nn.Module):
    def __init__(
        self,
        glove_weights: torch.Tensor,
        elmo_dim: int = 1024,
        hidden_size: int = 300,
        dropout: float = 0.5,
        freeze_glove: bool = True,
    ) -> None:
        super().__init__()
        self.glove = nn.Embedding.from_pretrained(
            glove_weights, freeze=freeze_glove, padding_idx=0
        )
        self.encoder = nn.LSTM(
            input_size=glove_weights.shape[1] + elmo_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(
        self,
        token_ids: torch.Tensor,
        elmo_embeddings: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        glove_embeddings = self.glove(token_ids)
        encoded, _ = self.encoder(torch.cat((glove_embeddings, elmo_embeddings), dim=-1))
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        target_states = encoded[batch_indices, target_indices]
        return self.classifier(self.dropout(target_states))
