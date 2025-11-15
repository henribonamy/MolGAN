"""Quick test to debug the draw function"""
import torch
from generator import MolGANGenerator
from config import DEVICE, NZ
from utils import draw, build_molecule, check_valid
from rdkit import Chem

# Load generator
generator = MolGANGenerator().to(DEVICE)
generator.load_state_dict(torch.load('generator.pt', map_location=DEVICE))
generator.eval()

# Generate one molecule
print("Generating a test molecule...")
with torch.no_grad():
    noise = torch.randn((1, NZ), device=DEVICE)
    fake_A, fake_X = generator.forward(noise)

    fake_X_sample = fake_X.squeeze(0).cpu().numpy()
    fake_A_sample = fake_A.squeeze(0).cpu().numpy()

print(f"X shape: {fake_X_sample.shape}")
print(f"A shape: {fake_A_sample.shape}")
print(f"X dtype: {fake_X_sample.dtype}")
print(f"A dtype: {fake_A_sample.dtype}")

# Check validity
is_valid = check_valid(fake_X_sample, fake_A_sample)
print(f"Is valid: {is_valid}")

# Try to build molecule
mol = build_molecule(fake_X_sample, fake_A_sample, sanitize=True)
print(f"Molecule object: {mol}")

if mol:
    print(f"Molecule has {mol.GetNumAtoms()} atoms")
    print(f"Molecule has {mol.GetNumBonds()} bonds")

    # Try to get SMILES
    try:
        smiles = Chem.MolToSmiles(mol)
        print(f"SMILES: {smiles}")
    except Exception as e:
        print(f"Failed to get SMILES: {e}")

    # Try to draw
    print("\nAttempting to draw...")
    success = draw(fake_X_sample, fake_A_sample, "test_molecule.png")
    print(f"Draw success: {success}")
else:
    print("ERROR: Could not build molecule!")
