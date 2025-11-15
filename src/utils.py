import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

ATOMS = ["C", "O", "N", "F"]

N = 9
T = 5
Y = 4

NZ = 32

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def get_bond_type(bond_vector):
    if not isinstance(bond_vector, np.ndarray):
        bond_vector = np.array(bond_vector)

    idx = np.argmax(bond_vector)

    if idx == 0:
        return None

    bond_types = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
    return bond_types[idx]


def build_molecule(X, A, sanitize=True):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    if X.ndim == 3 and X.shape[0] == 1:
        X = X.squeeze(0)

    mol = Chem.RWMol()

    original_to_mol_idx = {}
    for original_idx, row in enumerate(X):
        atom_idx = np.argmax(row)
        if atom_idx >= len(ATOMS):
            continue
        atom_type = ATOMS[atom_idx]
        atom = Chem.Atom(atom_type)
        mol_idx = mol.AddAtom(atom)
        original_to_mol_idx[original_idx] = mol_idx

    N = X.shape[0]

    for i in range(N):
        for j in range(i + 1, N):
            if i not in original_to_mol_idx or j not in original_to_mol_idx:
                continue

            bond_vec = A[i, j]
            bond_type = get_bond_type(bond_vec)
            if bond_type is not None:
                try:
                    mol.AddBond(original_to_mol_idx[i], original_to_mol_idx[j], bond_type)
                except Exception as e:
                    return None

    if sanitize:
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None
    else:
        return mol


def check_valid(X, A):
    mol = build_molecule(X, A, sanitize=True)
    return mol is not None


def draw(X, A, filename="image.png"):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    mol = build_molecule(X, A, sanitize=True)
    if mol:
        Draw.MolToFile(mol, filename, size=(300, 300))
        return True
    else:
        mol = build_molecule(X, A, sanitize=False)
        if mol:
            try:
                mol = mol.GetMol()
                Draw.MolToFile(mol, filename, size=(300, 300))
                return True
            except Exception as e:
                return False
    return False
