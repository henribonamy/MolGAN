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

def calculate_gradient_penalty(discriminator, real_A, real_X, fake_A, fake_X):
    batch_size = real_A.size(0)
    device = real_A.device

    # Sample eps for A and X (supports different shapes cleanly)
    eps_A = torch.rand(batch_size, 1, 1, 1, device=device)
    eps_X = torch.rand(batch_size, 1, 1, device=device)

    # Interpolate directly on the tensors the discriminator actually uses
    interp_A = eps_A * real_A + (1 - eps_A) * fake_A
    interp_X = eps_X * real_X + (1 - eps_X) * fake_X

    # Enable gradient tracking
    interp_A.requires_grad_(True)
    interp_X.requires_grad_(True)

    # Forward pass
    logits = discriminator(interp_A, interp_X)

    # Gradient of output wrt inputs
    grad_outputs = torch.ones_like(logits)

    gradients = autograd.grad(
        outputs=logits,
        inputs=[interp_A, interp_X],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    # Concatenate gradients of A and X
    grad_A, grad_X = gradients

    grad_A = grad_A.reshape(batch_size, -1)
    grad_X = grad_X.reshape(batch_size, -1)

    grad_total = torch.cat([grad_A, grad_X], dim=1)

    grad_norm = grad_total.norm(2, dim=1)

    # Raw gradient penalty (lambda will be applied outside)
    return ((grad_norm - 1) ** 2).mean()

def calculate_diversity_loss(fake_A, fake_X):
    """
    Calculate diversity loss to encourage the generator to produce diverse molecules.
    Returns the negative mean pairwise distance (so minimizing encourages diversity).

    Args:
        fake_A: Generated adjacency tensors (batch_size, N, N, bond_types)
        fake_X: Generated node features (batch_size, N, features)

    Returns:
        Diversity loss (negative of mean pairwise distance)
    """
    batch_size = fake_X.size(0)

    # Flatten the generated molecules to vectors
    X_flat = fake_X.reshape(batch_size, -1)
    A_flat = fake_A.reshape(batch_size, -1)
    generated_flat = torch.cat([X_flat, A_flat], dim=1)  # (batch_size, total_features)

    # Compute pairwise L2 distances
    # Expand dimensions for broadcasting
    gen_i = generated_flat.unsqueeze(1)  # (batch_size, 1, features)
    gen_j = generated_flat.unsqueeze(0)  # (1, batch_size, features)

    # Pairwise squared distances
    pairwise_distances = torch.sum((gen_i - gen_j) ** 2, dim=2)  # (batch_size, batch_size)

    # Only consider upper triangle (excluding diagonal) to avoid double counting
    mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1)
    distances = pairwise_distances * mask

    # Mean pairwise distance (only non-zero elements)
    num_pairs = (batch_size * (batch_size - 1)) / 2
    mean_distance = distances.sum() / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    # Return negative distance (so minimizing this loss maximizes diversity)
    return -mean_distance

def get_bond_type(bond_vector):
    """
    Converts a one-hot or soft bond vector to an RDKit BondType.
    bond_vector: [no bond, single, double, triple]
    """
    if not isinstance(bond_vector, np.ndarray):
        bond_vector = np.array(bond_vector)

    idx = np.argmax(bond_vector)

    # If the highest probability is for "no bond" (channel 0), return None
    if idx == 0:
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
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # Handle case where X has extra dimension (e.g., shape (1, N, features))
    if X.ndim == 3 and X.shape[0] == 1:
        X = X.squeeze(0)

    mol = Chem.RWMol()
    atom_indices = []

    # Add atoms and track which original indices map to actual atoms
    original_to_mol_idx = {}
    for original_idx, row in enumerate(X):
        atom_idx = np.argmax(row)
        # Skip padding atoms (index 4 is the padding atom)
        if atom_idx >= len(ATOMS):
            continue  # skip padding rows
        atom_type = ATOMS[atom_idx]
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


def training_checks(generator, num_samples, device, nz):
    """
    Sample molecules from the generator and check validity and uniqueness.

    Args:
        generator: The MolGAN generator model
        num_samples: Number of molecules to sample
        device: torch device to use
        nz: Dimension of the noise vector

    Returns:
        Tuple of (num_valid, num_unique) molecules out of num_samples
    """
    generator.eval()
    valid_count = 0
    valid_smiles = set()

    with torch.no_grad():
        for i in range(num_samples):
            # Generate noise vector
            noise = torch.randn((1, nz), device=device)
            # Generate molecule - returns (A, X)
            fake_A, fake_X = generator.forward(noise)

            # Remove batch dimension and move to CPU
            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            # Build molecule and check validity
            mol = build_molecule(fake_X_sample, fake_A_sample, sanitize=True)
            if mol is not None:
                valid_count += 1
                # Get canonical SMILES for uniqueness check
                try:
                    smiles = Chem.MolToSmiles(mol)
                    valid_smiles.add(smiles)
                except:
                    pass  # If SMILES conversion fails, skip uniqueness tracking

    generator.train()
    return valid_count, len(valid_smiles)


def draw(X, A, filename="image.png"):
    """Draw a molecule and save to file. Tries with sanitization first, falls back to unsanitized."""
    # Convert tensors to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # First try with sanitization for better bond display
    mol = build_molecule(X, A, sanitize=True)
    if mol:
        Draw.MolToFile(mol, filename, size=(300, 300))
        return True
    else:
        # Fall back to unsanitized if sanitization fails
        mol = build_molecule(X, A, sanitize=False)
        if mol:
            try:
                mol = mol.GetMol()  # Convert RWMol to Mol for drawing
                Draw.MolToFile(mol, filename, size=(300, 300))
                return True
            except Exception as e:
                print(f"Warning: Failed to draw molecule to {filename}: {e}")
                return False
    return False


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
