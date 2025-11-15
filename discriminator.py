import torch
import torch.nn as nn

class MolGANDiscriminator(nn.Module):
    def __init__(self, node_feat_dim=5, num_rels=4, hidden_dims=[64, 32], agg_dim=128):
        super().__init__()
        dims = [node_feat_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.layers.append(RelationalGCNLayer(dims[i], dims[i+1], num_rels, node_feat_dim))

        last_hidden = hidden_dims[-1]
        self.i_mlp = nn.Linear(last_hidden + node_feat_dim, 1)
        self.j_mlp = nn.Linear(last_hidden + node_feat_dim, agg_dim)

        self.fc1 = nn.Linear(agg_dim, agg_dim)
        self.fc2 = nn.Linear(agg_dim, 1)

    def forward(self, a, x):
        # a: [B, N, N, 4] where channels are [no-bond, single, double, triple]
        # Use all 4 channels for gradient flow, but only bond channels (1-3) for connectivity
        a_rel = a  # [B, N, N, 4] - now using all channels
        # compute continuous adjacency by summing only the bond channels (exclude no-bond)
        adj = a[..., 1:].sum(dim=-1)    # [B, N, N], only sum bond channels
        degrees = adj.sum(dim=-1)        # [B, N], differentiable

        h = x
        for layer in self.layers:
            h = layer(h, x, a_rel, degrees)
        
        concat = torch.cat([h, x], dim=-1)        # [B, N, H+F]
        gate = torch.sigmoid(self.i_mlp(concat))  # [B, N, 1]
        value = torch.tanh(self.j_mlp(concat))    # [B, N, agg_dim]
        h_g_prime = (gate * value).sum(dim=1)     # [B, agg_dim]
        h_g = torch.tanh(h_g_prime)               # [B, agg_dim]
        
        out = torch.tanh(self.fc1(h_g))
        out = self.fc2(out)                       # [B, 1]
        return out

class RelationalGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, node_feat_dim):
        super().__init__()
        self.f_s = nn.Linear(in_feat + node_feat_dim, out_feat)
        self.f_y = nn.ModuleList([nn.Linear(in_feat + node_feat_dim, out_feat) for _ in range(num_rels)])

    def forward(self, h, x, a_rel, degrees, eps=1e-6):
        # h: [B, N, in_feat]   x: [B, N, node_feat_dim]
        concat = torch.cat([h, x], dim=-1)             # [B, N, in+F]
        self_part = self.f_s(concat)                   # [B, N, out]
        
        neighbor = 0
        # a_rel[..., y] is [B, N, N]; temp [B, N, out]; matmul gives [B, N, out]
        for r, linear in enumerate(self.f_y):
            temp = linear(concat)                      # [B, N, out]
            a_y = a_rel[..., r]                        # [B, N, N] (float, differentiable)
            # multiply adjacency with neighbor features: a_y @ temp
            sum_j = torch.matmul(a_y, temp)            # [B, N, out]
            neighbor = neighbor + sum_j
        
        # degrees: [B, N], make safe denominator and keep differentiable
        denom = degrees.unsqueeze(-1) + eps            # [B, N, 1]
        neighbor = neighbor / denom                    # [B, N, out]
        
        h_prime = self_part + neighbor
        return torch.tanh(h_prime)

