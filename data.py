import torch
import os
import torch_geometric.datasets
from utils import N, T, Y

def load_data():
    raw_data = torch_geometric.datasets.QM9(root=os.getcwd())
    return raw_data

def raw_to_XA(mol: torch_geometric.data.data.Data):
    atom_to_index = {6: 0, 8: 1, 7: 2, 9: 3}
    bond_map = {0: 1, 1: 2, 2: 3, 3: 2}

    atom_types = mol.z.tolist()
    non_h_indices = [i for i, z in enumerate(atom_types) if z != 1]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    n = len(non_h_indices)

    X = torch.zeros((N, T))
    for new_idx, old_idx in enumerate(non_h_indices):
        atomic_num = atom_types[old_idx]
        if atomic_num in atom_to_index:
            X[new_idx, atom_to_index[atomic_num]] = 1
        else:
            raise ValueError(f"Unsupported atom type: {atomic_num}")
    for pad_idx in range(n, N):
        X[pad_idx, 4] = 1

    A = torch.zeros((n, n, Y))
    A[:, :, 0] = 1

    edge_index = mol.edge_index
    edge_attr = mol.edge_attr
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src in index_map and dst in index_map:
            new_src = index_map[src]
            new_dst = index_map[dst]
            bond_vector = edge_attr[i].tolist()
            bond_type = bond_vector.index(1)
            channel = bond_map.get(bond_type)
            if channel is None:
                raise ValueError(f"Invalid bond type encoding: {bond_vector}")
            A[new_src, new_dst, 0] = 0
            A[new_src, new_dst, channel] = 1
            A[new_dst, new_src, 0] = 0
            A[new_dst, new_src, channel] = 1

    A_padded = torch.zeros((N, N, Y))
    A_padded[:, :, 0] = 1
    A_padded[:n, :n, :] = A
    return X, A_padded
