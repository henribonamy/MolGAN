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
        a_rel = a
        adj = a[..., 1:].sum(dim=-1)
        degrees = adj.sum(dim=-1)

        h = x
        for layer in self.layers:
            h = layer(h, x, a_rel, degrees)

        concat = torch.cat([h, x], dim=-1)
        gate = torch.sigmoid(self.i_mlp(concat))
        value = torch.tanh(self.j_mlp(concat))
        h_g_prime = (gate * value).sum(dim=1)
        h_g = torch.tanh(h_g_prime)

        out = torch.tanh(self.fc1(h_g))
        out = self.fc2(out)
        return out

class RelationalGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, node_feat_dim):
        super().__init__()
        self.f_s = nn.Linear(in_feat + node_feat_dim, out_feat)
        self.f_y = nn.ModuleList([nn.Linear(in_feat + node_feat_dim, out_feat) for _ in range(num_rels)])

    def forward(self, h, x, a_rel, degrees, eps=1e-6):
        concat = torch.cat([h, x], dim=-1)
        self_part = self.f_s(concat)

        neighbor = 0
        for r, linear in enumerate(self.f_y):
            temp = linear(concat)
            a_y = a_rel[..., r]
            sum_j = torch.matmul(a_y, temp)
            neighbor = neighbor + sum_j

        denom = degrees.unsqueeze(-1) + eps
        neighbor = neighbor / denom

        h_prime = self_part + neighbor
        return torch.tanh(h_prime)
