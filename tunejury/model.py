"""TuneJury: pairwise music-preference reward model.

The TuneJury head is a 4-layer MLP over frozen audio + text encoder
embeddings. The released checkpoint concatenates LAION-CLAP-Music
(512-d audio + 512-d text) with MERT-v1-330M (1024-d audio) for a
2048-d input. ~2.8M trainable parameters; encoders are frozen at
inference and contribute ~515M frozen parameters with no gradient
flow (paper §3).

At training time the same MLP, with weights shared across the two
clips of a preference pair (A, B), produces s(A) and s(B) for the
pairwise logistic loss
    L = -log sigma(s(A) - s(B)).

At inference the head scores one clip per forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TuneJury(nn.Module):
    """Shared-weight pairwise scoring head.

    Parameters
    ----------
    input_dim : int
        Concatenated embedding dimension. Default 2048 (the released
        CLAP+MERT instantiation).
    hidden_layers : list[int]
        Hidden layer widths. Default [1024, 512, 256, 128].
    dropout_rate : float
        Dropout probability between hidden layers. Default 0.5.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [1024, 512, 256, 128]

        last_dim = input_dim
        layers: list[nn.Module] = []
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score a single clip x_a, or return s(A) - s(B) for a pair."""
        if x_b is not None:
            return self.net(x_a) - self.net(x_b)
        return self.net(x_a)
