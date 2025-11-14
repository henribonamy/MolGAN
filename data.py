import torch
import os
import torch_geometric.datasets
from config import N, T, Y

def load_data():
    raw_data = torch_geometric.datasets.QM9(root=os.getcwd())
    return raw_data

def raw_to_XA(mol: torch_geometric.data.data.Data):
    """Convert PyG molecular graph to (X, A) format excluding hydrogens (atomic_num = 1)."""
    atom_to_index = {6: 0, 8: 1, 7: 2, 9: 3}  # C, O, N, F
    bond_map = {0: 1, 1: 2, 2: 3, 3: 2}  # single→1, double→2, triple→3, aromatic→2 (approx)
    
    atom_types = mol.z.tolist()
    # Map non-H atoms to new indices
    non_h_indices = [i for i, z in enumerate(atom_types) if z != 1]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    n = len(non_h_indices)
    
    # Initialize X (node features) with zeros
    X = torch.zeros((N, T))
    # Set actual atom types
    for new_idx, old_idx in enumerate(non_h_indices):
        atomic_num = atom_types[old_idx]
        if atomic_num in atom_to_index:
            X[new_idx, atom_to_index[atomic_num]] = 1
        else:
            raise ValueError(f"Unsupported atom type: {atomic_num}")
    # Set padding atoms to one-hot padding vector [0, 0, 0, 0, 1] (assuming index 4)
    for pad_idx in range(n, N):
        X[pad_idx, 4] = 1  # Padding atom index is 4
    
    # Initialize A (adjacency tensor) with no-bond everywhere (channel 0 = 1)
    A = torch.zeros((n, n, Y))
    A[:, :, 0] = 1  # Default: no-bond for all pairs
    
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src in index_map and dst in index_map:
            new_src = index_map[src]
            new_dst = index_map[dst]
            # edge_attr[i] is one-hot: [single, double, triple, aromatic]
            bond_vector = edge_attr[i].tolist()
            bond_type = bond_vector.index(1)
            if bond_type == 3:
                print("Warning: Aromatic bond found. Mapping to double bond.")
            channel = bond_map.get(bond_type)
            if channel is None:
                raise ValueError(f"Invalid bond type encoding: {bond_vector}")
            # Override no-bond with the actual bond type
            A[new_src, new_dst, 0] = 0
            A[new_src, new_dst, channel] = 1
            # Since undirected, set both directions (though QM9 already has both)
            A[new_dst, new_src, 0] = 0
            A[new_dst, new_src, channel] = 1
    
    # Pad A to full shape, with no-bond everywhere by default
    A_padded = torch.zeros((N, N, Y))
    A_padded[:, :, 0] = 1  # Default: no-bond for all pairs (including padded)
    A_padded[:n, :n, :] = A  # Overwrite real graph portion
    return X, A_padded
