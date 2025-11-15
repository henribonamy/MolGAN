# MolGAN

A PyTorch implementation of the MolGAN paper, used to generate valid molecular structures. The paper uses RL to optimize for certain chemical properties, but this implementation does without RL by monitoring validity and novelty algorithmically.

## Generated Molecules

<p align="center">
  <img src="media/novel_molecule_2.png" width="200"/>
  <img src="media/novel_molecule_3.png" width="200"/>
  <img src="media/novel_molecule_5.png" width="200"/>
</p>

## Architecture

This project trains a generative model to create novel, chemically valid molecules represented as graphs. The generator produces adjacency matrices (bonds) and node features (atoms), while a graph convolutional discriminator evaluates their realism.

- **Generator**: Multi-layer perceptron with Gumbel-Softmax for discrete graph generation
- **Discriminator**: Relational Graph Convolutional Network (R-GCN) with graph pooling
- **Training**: WGAN-GP with gradient penalty for (relatively) stable adversarial learning
- **Dataset**: QM9 molecular dataset (filtered to C, O, N, F atoms, up to 9 heavy atoms)

## Training

Run the training script:

```bash
python train.py
```

This will:
- Download and prepare the QM9 dataset
- Train the generator and discriminator for 15 epochs
- Save the best (for % validity) generator checkpoint as `generator_<validity>.pt`

### Example Training Progress

<p align="center">
  <img src="media/training_progress.png" width="600"/>
</p>

## Sampling Molecules

Generate novel molecules using the trained model (you can use the provided `generator.pt` or train your own.)

```bash
python sample_molecules.py
```

This will:
- Load the trained generator from `generator.pt`
- Generate 20 novel molecules (not in training set)
- Save images to `sampled_molecules/`
- Ensure all molecules are valid, unique, and novel

## Original Paper

Based on:
- De Cao, N., & Kipf, T. (2018). MolGAN: An implicit generative model for small molecular graphs.
