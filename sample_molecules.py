import torch
from pathlib import Path
from rdkit import RDLogger
import numpy as np

from generator import MolGANGenerator
from utils import DEVICE, NZ, check_valid, draw
from data import load_data, raw_to_XA

NUM_MOLECULES_TO_GENERATE = 20
OUTPUT_DIR = "sampled_molecules"
GENERATOR_PATH = "generator.pt"

RDLogger.DisableLog('rdApp.*')


def matrices_equal(X1, A1, X2, A2):
    return np.array_equal(X1, X2) and np.array_equal(A1, A2)


def get_dataset_molecules():
    data = load_data()
    dataset_molecules = []

    for mol_data in data:
        try:
            X, A = raw_to_XA(mol_data)
            X_np = X.cpu().numpy()
            A_np = A.cpu().numpy()
            dataset_molecules.append((X_np, A_np))
        except Exception as e:
            continue

    return dataset_molecules


def is_in_dataset(X, A, dataset_molecules):
    for X_dataset, A_dataset in dataset_molecules:
        if matrices_equal(X, A, X_dataset, A_dataset):
            return True
    return False


def is_duplicate(X, A, generated_molecules):
    for X_gen, A_gen in generated_molecules:
        if matrices_equal(X, A, X_gen, A_gen):
            return True
    return False


def sample_novel_molecules(generator, dataset_molecules, num_samples, device, nz, output_dir):
    generator.eval()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    generated_molecules = []
    attempts = 0
    max_attempts = num_samples * 1000

    valid_count = 0
    in_dataset_count = 0
    duplicate_count = 0

    with torch.no_grad():
        while len(generated_molecules) < num_samples and attempts < max_attempts:
            attempts += 1

            noise = torch.randn((1, nz), device=device)
            fake_A, fake_X = generator.forward(noise)

            fake_X_sample = fake_X.squeeze(0).cpu().numpy()
            fake_A_sample = fake_A.squeeze(0).cpu().numpy()

            if check_valid(fake_X_sample, fake_A_sample):
                valid_count += 1

                if is_in_dataset(fake_X_sample, fake_A_sample, dataset_molecules):
                    in_dataset_count += 1
                    continue

                if is_duplicate(fake_X_sample, fake_A_sample, generated_molecules):
                    duplicate_count += 1
                    continue

                generated_molecules.append((fake_X_sample, fake_A_sample))

                mol_idx = len(generated_molecules)
                filename = output_path / f"novel_molecule_{mol_idx}.png"
                draw(fake_X_sample, fake_A_sample, str(filename))
                print(f"[{mol_idx}/{num_samples}] Generated novel molecule -> {filename}")

    if len(generated_molecules) < num_samples:
        print(f"\nWarning: Only found {len(generated_molecules)} novel molecules after {attempts} attempts.")

    generator.train()
    return generated_molecules


def main():
    print("=" * 70)
    print("NOVEL MOLECULE SAMPLER")
    print("=" * 70)

    generator = MolGANGenerator().to(DEVICE)
    checkpoint_path = Path(GENERATOR_PATH)

    if not checkpoint_path.exists():
        print(f"ERROR: Generator checkpoint not found at '{GENERATOR_PATH}'")
        return

    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"Loaded generator from {checkpoint_path}\n")

    dataset_molecules = get_dataset_molecules()
    print(f"Loaded {len(dataset_molecules)} molecules from training dataset\n")

    novel_molecules = sample_novel_molecules(
        generator=generator,
        dataset_molecules=dataset_molecules,
        num_samples=NUM_MOLECULES_TO_GENERATE,
        device=DEVICE,
        nz=NZ,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print(f"Generated {len(novel_molecules)} novel molecules")
    print(f"Images saved to: ./{OUTPUT_DIR}/")
    print("=" * 70)

    RDLogger.EnableLog('rdApp.*')


if __name__ == "__main__":
    main()
