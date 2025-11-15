import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NZ, N, T, Y


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
            nn.Linear(h3, N * T + N * N * Y),
        )

    def forward(self, z):
        batch_size = z.size(0)
        out = self.net(z)

        X_logits = out[:, : N * T]
        A_logits = out[:, N * T :]

        X_logits = X_logits.view(batch_size, N, T)
        A_logits = A_logits.view(batch_size, N, N, Y)

        X = F.gumbel_softmax(X_logits, dim=-1, tau=1.0, hard=False)
        A = F.gumbel_softmax(A_logits, dim=-1, tau=1.0, hard=False)
        return A, X
