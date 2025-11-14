import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NZ, N, T, Y


class MolGANGenerator(nn.Module):
    def __init__(self, hidden_dims=(128, 256, 512)):
        super().__init__()

        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(NZ, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, N * T + N * N * Y),  # → 9*5 + 9*9*4 = 369
        )

    def forward(self, z):
        """ Takes z ~ N(0;1) and returns X, A)"""
        batch_size = z.size(0)
        out = self.net(z)  # (batch_size, 369)

        X_logits = out[:, : N * T]  # (batch_size, 45)
        A_logits = out[:, N * T :]  # (batch_size, 324)

        # Reshape
        X_logits = X_logits.view(batch_size, N, T)  # (batch_size, 9, 5)
        A_logits = A_logits.view(batch_size, N, N, Y)  # (batch_size, 9, 9, 4)

        X = F.gumbel_softmax(X_logits, dim=-1, tau=0.1, hard=True)
        A = F.gumbel_softmax(A_logits, dim=-1, tau=0.1, hard=True)
        return A, X
