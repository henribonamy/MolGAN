import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType

from config import ATOM_EQUIV, ATOMS

def calculate_gradient_penalty(discriminator, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # 9*5 + 9*9*4
        interp_logits = discriminator.forward(interpolation[:, 45:].reshape(BATCH_SIZE, 9, 9, 4), interpolation[:, :45].reshape(BATCH_SIZE, 9, 5))

        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

def get_bond_type(bond_vector):
    """
    Converts a one-hot bond vector to an RDKit BondType.
    bond_vector: [no bond, single, double, triple]
    """
    if not isinstance(bond_vector, np.ndarray):
        bond_vector = np.array(bond_vector)

    idx = np.argmax(bond_vector)

    if idx == 0 or bond_vector[idx] == 0:
        return None  # no bond

    # Map to RDKit bond type
    bond_types = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
    return bond_types[idx]


def build_molecule(X, A, sanitize=True):
    """
    Converts MolGAN-style (X, A) representation to an RDKit molecule.

    X: Atom features (one-hot) for heavy atoms [C, O, N, F]
    A: Adjacency tensor (N x N x 4) with bond type one-hot: [no, single, double, triple]
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    # Handle case where X has extra dimension (e.g., shape (1, N, features))
    if X.ndim == 3 and X.shape[0] == 1:
        X = X.squeeze(0)

    mol = Chem.RWMol()
    atom_indices = []

    # Add atoms and track which original indices map to actual atoms
    original_to_mol_idx = {}
    for original_idx, row in enumerate(X):
        if row[-1] == 1:
            continue  # skip padding rows
        atom_type = ATOMS[np.argmax(row)]
        atom = Chem.Atom(atom_type)
        mol_idx = mol.AddAtom(atom)
        original_to_mol_idx[original_idx] = mol_idx
        atom_indices.append(mol_idx)

    N = X.shape[0]

    # Add bonds using the mapped indices
    for i in range(N):
        for j in range(i + 1, N):
            # Skip if either atom is padding
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
            mol = Chem.AddHs(mol)  # infer hydrogens
            return mol
        except Exception:
            return None
    else:
        return mol


def check_valid(X, A):
    mol = build_molecule(X, A, sanitize=True)
    return mol is not None


def sample_and_check_validity(generator, num_samples, device, nz):
    """
    Sample molecules from the generator and check how many are chemically valid.

    Args:
        generator: The MolGAN generator model
        num_samples: Number of molecules to sample
        device: torch device to use
        nz: Dimension of the noise vector

    Returns:
        Number of valid molecules out of num_samples
    """
    generator.eval()
    valid_count = 0

    with torch.no_grad():
        for i in range(num_samples):
            # Generate noise vector
            noise = torch.randn((1, nz), device=device)
            # Generate molecule - returns (A, X)
            fake_A, fake_X = generator.forward(noise)

            # Remove batch dimension and move to CPU
            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            # Check validity - note check_valid expects (X, A) not (A, X)
            if check_valid(fake_X_sample, fake_A_sample):
                valid_count += 1

    generator.train()
    return valid_count


def draw(X, A, filename="image.png"):
    mol = build_molecule(X, A, sanitize=False)
    if mol:
        Draw.MolToFile(mol, filename)


def print_mol(atoms: torch.Tensor):
    return [ATOM_EQUIV[index.item()] for index in atoms.squeeze()]


def generate_and_display_molecules(num_molecules=10, output_dir="generated_molecules"):
    """
    Generate molecules from the trained generator, save them as images, and display them.
    Loads the generator from 'generator.pt' file.

    Args:
        num_molecules: Number of molecules to generate (default: 10)
        output_dir: Directory to save the images (default: "generated_molecules")

    Returns:
        List of PIL Image objects and validity info
    """
    from PIL import Image
    from generator import MolGANGenerator
    from config import DEVICE, NZ

    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # Load generator
    generator = MolGANGenerator().to(DEVICE)
    checkpoint_path = Path('generator.pt')

    if checkpoint_path.exists():
        print(f"Loading trained generator from {checkpoint_path}")
        generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Generator loaded successfully!\n")
    else:
        print("Warning: No trained generator found at 'generator.pt'. Using randomly initialized generator.\n")

    generator.eval()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    molecule_info = []
    images = []

    print(f"Generating {num_molecules} molecules...")
    with torch.no_grad():
        for i in range(num_molecules):
            # Generate noise vector
            noise = torch.randn((1, NZ), device=DEVICE)
            # Generate molecule - returns (A, X)
            fake_A, fake_X = generator.forward(noise)

            # Remove batch dimension and move to CPU
            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            # Check if valid
            is_valid = check_valid(fake_X_sample, fake_A_sample)

            # Save the molecule image
            suffix = "valid" if is_valid else "invalid"
            filename = output_path / f"molecule_{i+1}_{suffix}.png"
            draw(fake_X_sample, fake_A_sample, str(filename))

            # Load image for display
            img = Image.open(filename)
            images.append(img)

            status = "✓ VALID" if is_valid else "✗ INVALID"
            molecule_info.append((str(filename), is_valid))
            print(f"  {status}: {filename}")

    valid_count = sum(1 for _, is_valid in molecule_info if is_valid)
    print(f"\nSummary: {valid_count}/{num_molecules} molecules are chemically valid ({valid_count*10}%)")
    print(f"Images saved to: ./{output_dir}/\n")

    # Re-enable RDKit logging
    RDLogger.EnableLog('rdApp.*')

    # Display images (return for notebook/display)
    return images, molecule_info


# def repr_molecule(mol : torch_geometric.data.data.Data):
#     print(f"Repr shape : {mol.z.size()[0]}x5")
#     print(mol_equiv(mol.z))
#
#     links = {}
#     for i, element in enumerate(mol.edge_index[0]):
#         element = element.item()
#         if element in links.keys():
#             links[element].append(mol.edge_index[1][i].item())
#         else:
#             links[element] = [mol.edge_index[1][i].item()]
#
#     for key, values in links.items():
#         print(f"Atom {key} has links with {values}")
