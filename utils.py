import torch
import torch.autograd as autograd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType

ATOMS = ["C", "O", "N", "F"]
ATOM_EQUIV = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F"}

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

def calculate_gradient_penalty(discriminator, real_A, real_X, fake_A, fake_X):
    batch_size = real_A.size(0)
    device = real_A.device

    eps_A = torch.rand(batch_size, 1, 1, 1, device=device)
    eps_X = torch.rand(batch_size, 1, 1, device=device)

    interp_A = eps_A * real_A + (1 - eps_A) * fake_A
    interp_X = eps_X * real_X + (1 - eps_X) * fake_X

    interp_A.requires_grad_(True)
    interp_X.requires_grad_(True)

    logits = discriminator(interp_A, interp_X)

    grad_outputs = torch.ones_like(logits)

    gradients = autograd.grad(
        outputs=logits,
        inputs=[interp_A, interp_X],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    grad_A, grad_X = gradients

    grad_A = grad_A.reshape(batch_size, -1)
    grad_X = grad_X.reshape(batch_size, -1)

    grad_total = torch.cat([grad_A, grad_X], dim=1)

    grad_norm = grad_total.norm(2, dim=1)

    return ((grad_norm - 1) ** 2).mean()

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
    atom_indices = []

    original_to_mol_idx = {}
    for original_idx, row in enumerate(X):
        atom_idx = np.argmax(row)
        if atom_idx >= len(ATOMS):
            continue
        atom_type = ATOMS[atom_idx]
        atom = Chem.Atom(atom_type)
        mol_idx = mol.AddAtom(atom)
        original_to_mol_idx[original_idx] = mol_idx
        atom_indices.append(mol_idx)

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


def training_checks(generator, num_samples, device, nz):
    generator.eval()
    valid_count = 0
    valid_smiles = set()

    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn((1, nz), device=device)
            fake_A, fake_X = generator.forward(noise)

            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            mol = build_molecule(fake_X_sample, fake_A_sample, sanitize=True)
            if mol is not None:
                valid_count += 1
                try:
                    smiles = Chem.MolToSmiles(mol)
                    valid_smiles.add(smiles)
                except:
                    pass

    generator.train()
    return valid_count, len(valid_smiles)


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
