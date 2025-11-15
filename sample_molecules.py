"""
Sample unique molecules from the trained generator that are not in the training dataset.
"""
import torch
from pathlib import Path
from rdkit import RDLogger
import numpy as np

from generator import MolGANGenerator
from config import DEVICE, NZ
from utils import check_valid, draw
from data import load_data, raw_to_XA

# ========== CONFIGURATION ==========
NUM_MOLECULES_TO_GENERATE = 20  # Number of unique, novel molecules to generate
OUTPUT_DIR = "sampled_molecules"
GENERATOR_PATH = "generator.pt"
# ===================================

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def matrices_equal(X1, A1, X2, A2):
    """Check if two molecules represented as (X, A) matrices are identical."""
    return np.array_equal(X1, X2) and np.array_equal(A1, A2)


def get_dataset_molecules():
    """
    Load the training dataset and convert to (X, A) format.
    Returns a list of (X, A) tuples.
    """
    print("Loading training dataset...")
    data = load_data()
    dataset_molecules = []

    print(f"Converting {len(data)} molecules to (X, A) format...")
    for i, mol_data in enumerate(data):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(data)} molecules...")

        try:
            X, A = raw_to_XA(mol_data)
            X_np = X.cpu().numpy()
            A_np = A.cpu().numpy()
            dataset_molecules.append((X_np, A_np))
        except Exception as e:
            # Skip molecules that fail conversion
            continue

    print(f"Dataset contains {len(dataset_molecules)} molecules.\n")
    return dataset_molecules


def is_in_dataset(X, A, dataset_molecules):
    """Check if a molecule (X, A) is in the dataset."""
    for X_dataset, A_dataset in dataset_molecules:
        if matrices_equal(X, A, X_dataset, A_dataset):
            return True
    return False


def is_duplicate(X, A, generated_molecules):
    """Check if a molecule (X, A) is already in the generated list."""
    for X_gen, A_gen in generated_molecules:
        if matrices_equal(X, A, X_gen, A_gen):
            return True
    return False


def sample_novel_molecules(generator, dataset_molecules, num_samples, device, nz, output_dir):
    """
    Sample molecules from the generator that are:
    1. Chemically valid
    2. Not in the training dataset
    3. Unique from each other

    Args:
        generator: The trained MolGAN generator
        dataset_molecules: List of (X, A) tuples from the training dataset
        num_samples: Number of novel molecules to generate
        device: torch device to use
        nz: Dimension of the noise vector
        output_dir: Directory to save molecule images

    Returns:
        List of (X, A) tuples for the generated molecules
    """
    generator.eval()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    generated_molecules = []
    attempts = 0
    max_attempts = num_samples * 1000 # Prevent infinite loops

    print(f"Generating {num_samples} novel molecules...")
    print(f"(This may take a while as we filter out invalid and duplicate molecules)\n")

    valid_count = 0
    in_dataset_count = 0
    duplicate_count = 0

    with torch.no_grad():
        while len(generated_molecules) < num_samples and attempts < max_attempts:
            attempts += 1

            # Generate noise vector
            noise = torch.randn((1, nz), device=device)
            # Generate molecule - returns (A, X)
            fake_A, fake_X = generator.forward(noise)

            # Remove batch dimension and move to CPU
            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            # Check validity
            if check_valid(fake_X_sample, fake_A_sample):
                valid_count += 1

                # Check if in dataset
                if is_in_dataset(fake_X_sample, fake_A_sample, dataset_molecules):
                    in_dataset_count += 1
                    continue

                # Check if duplicate
                if is_duplicate(fake_X_sample, fake_A_sample, generated_molecules):
                    duplicate_count += 1
                    continue

                # Novel molecule found!
                generated_molecules.append((fake_X_sample, fake_A_sample))

                # Save the molecule image
                mol_idx = len(generated_molecules)
                filename = output_path / f"novel_molecule_{mol_idx}.png"
                success = draw(fake_X_sample, fake_A_sample, str(filename))

                if success:
                    print(f"  [{mol_idx}/{num_samples}] Generated novel molecule")
                    print(f"       Saved to: {filename}")
                else:
                    print(f"  [{mol_idx}/{num_samples}] Generated novel molecule but FAILED to draw image!")
                    print(f"       Attempted to save to: {filename}")

            # Progress update every 1000 attempts
            if attempts % 1000 == 0:
                invalid_count = attempts - valid_count
                print(f"\n  Stats after {attempts} attempts:")
                print(f"    - Valid: {valid_count} ({valid_count*100//attempts}%)")
                print(f"    - Invalid: {invalid_count} ({invalid_count*100//attempts}%)")
                print(f"    - In dataset: {in_dataset_count}")
                print(f"    - Duplicates: {duplicate_count}")
                print(f"    - Novel molecules found: {len(generated_molecules)}\n")

    if len(generated_molecules) < num_samples:
        print(f"\nWarning: Only found {len(generated_molecules)} novel molecules after {attempts} attempts.")
        print(f"Consider training the generator longer or increasing max_attempts.")

    generator.train()
    return generated_molecules


def main():
    print("=" * 70)
    print("NOVEL MOLECULE SAMPLER")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Target molecules: {NUM_MOLECULES_TO_GENERATE}")
    print(f"  - Generator path: {GENERATOR_PATH}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Device: {DEVICE}")
    print("=" * 70)
    print()

    # Load generator
    generator = MolGANGenerator().to(DEVICE)
    checkpoint_path = Path(GENERATOR_PATH)

    if not checkpoint_path.exists():
        print(f"ERROR: Generator checkpoint not found at '{GENERATOR_PATH}'")
        print("Please train a generator first using main_wgan.py")
        return

    print(f"Loading trained generator from {checkpoint_path}...")
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print("Generator loaded successfully!\n")

    # Load dataset molecules for comparison
    dataset_molecules = get_dataset_molecules()

    # Sample novel molecules
    novel_molecules = sample_novel_molecules(
        generator=generator,
        dataset_molecules=dataset_molecules,
        num_samples=NUM_MOLECULES_TO_GENERATE,
        device=DEVICE,
        nz=NZ,
        output_dir=OUTPUT_DIR
    )

    # Summary
    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Generated {len(novel_molecules)} novel molecules")
    print(f"Images saved to: ./{OUTPUT_DIR}/")
    print(f"\nAll molecules are:")
    print(f"  ✓ Chemically valid")
    print(f"  ✓ Not in the training dataset (based on exact matrix comparison)")
    print(f"  ✓ Unique from each other")
    print("=" * 70)

    # Re-enable RDKit logging
    RDLogger.EnableLog('rdApp.*')


if __name__ == "__main__":
    main()
