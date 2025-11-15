import os
import time

import torch
import torch.autograd as autograd
from rdkit import Chem, RDLogger

from data import load_data, raw_to_XA
from discriminator import MolGANDiscriminator
from generator import MolGANGenerator
from utils import DEVICE, NZ, build_molecule

RDLogger.DisableLog("rdApp.*")

BATCH_SIZE = 128
LAMBDA = 10
EPOCHS = 15
LR_G = 0.001
LR_D = 0.001


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
        only_inputs=True,
    )

    grad_A, grad_X = gradients

    grad_A = grad_A.reshape(batch_size, -1)
    grad_X = grad_X.reshape(batch_size, -1)

    grad_total = torch.cat([grad_A, grad_X], dim=1)

    grad_norm = grad_total.norm(2, dim=1)

    return ((grad_norm - 1) ** 2).mean()


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


def prepare_data():
    data = load_data()
    print("Transforming data...")
    transformed_data = [raw_to_XA(x) for x in data]
    print("Data transformed.")
    data_loader = torch.utils.data.DataLoader(
        transformed_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    return data_loader


def train(
    data_loader: torch.utils.data.DataLoader,
    generator: MolGANGenerator,
    discriminator: MolGANDiscriminator,
    epochs,
    lrG,
    lrD,
):
    print(f"[!] Using device {DEVICE} for training.")
    previous_valid = 0
    losses_G = []
    losses_D = []
    validity_metrics = []
    start_time = time.time()

    optimizerG = torch.optim.Adam(
        params=generator.parameters(), lr=lrG, betas=(0.0, 0.9)
    )
    optimizerD = torch.optim.Adam(
        params=discriminator.parameters(), lr=lrD, betas=(0.0, 0.9)
    )

    one = torch.tensor(1, dtype=torch.float, device=DEVICE)
    m_one = -1 * one

    for epoch in range(epochs):
        counter = 0
        for X, A in data_loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            batch_size = X.size(0)

            for _ in range(3):
                discriminator.zero_grad()

                D_real = discriminator.forward(A, X).squeeze(1)
                D_loss_real = D_real.mean()

                noise = torch.randn((batch_size, NZ), device=DEVICE)
                fake_A, fake_X = generator.forward(noise)
                D_fake = discriminator(fake_A.detach(), fake_X.detach()).squeeze(1)
                D_loss_fake = D_fake.mean()

                gradient_penalty = calculate_gradient_penalty(
                    discriminator, A, X, fake_A, fake_X
                )

                D_loss = D_loss_fake - D_loss_real + LAMBDA * gradient_penalty
                D_loss.backward()
                optimizerD.step()

            wasserstein_loss = D_loss_real - D_loss_fake

            generator.zero_grad()
            noise = torch.randn((batch_size, NZ), device=DEVICE)
            fake_A, fake_X = generator.forward(noise)

            G_loss = discriminator(fake_A, fake_X).squeeze(1)
            G_loss = G_loss.mean()
            G_loss.backward(m_one)

            optimizerG.step()

            if counter % (BATCH_SIZE * 200) == 0:
                print(
                    f"[{epoch}/{epochs}] (Iteration {counter}) \t Wasserstein distance: {wasserstein_loss.data:.4f}\t Penalty: {LAMBDA * gradient_penalty:.2f}\t Loss D: {D_loss:.4f}\t Loss G : {G_loss:.4f}\t Loss D real: {D_loss_real:.4f}\t Loss D fake: {D_loss_fake:.4f}\t ---- Dt : {time.time() - start_time:.2f}s."
                )

                num_valid, num_unique = training_checks(generator, 100, DEVICE, NZ)
                print(
                    f"Valid molecules: {num_valid}% | Unique molecules: {num_unique}%"
                )
                validity_metrics.append(num_valid)
                losses_G.append(G_loss)
                losses_D.append(D_loss)
                start_time = time.time()
                if num_valid >= previous_valid:
                    if previous_valid > 0:
                        os.remove(f"generator_{previous_valid}.pt")
                        torch.save(generator.state_dict(), f"generator_{num_valid}.pt")
                    previous_valid = num_valid
            counter += len(X)

    return losses_G, losses_D, validity_metrics


data_loader = prepare_data()
generator = MolGANGenerator().to(DEVICE)
discriminator = MolGANDiscriminator().to(DEVICE)

losses_G, losses_D, validity_metrics = train(
    data_loader, generator, discriminator, EPOCHS, LR_G, LR_D
)
