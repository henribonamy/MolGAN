import torch

# CHEMICAL CONSTANTS
ATOMS = ["C", "O", "N", "F"]
ATOM_EQUIV = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F"}

N = 9  # nombre max de nœuds
T = 5  # nombre de types d’atomes
Y = 4  # nombre de types de liaison


# TRAINING CONSTANTS
NZ = 32  # Z size
BATCH_SIZE = 32
NUM_WORKERS = 4

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
