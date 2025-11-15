import os
import time

import torch
from rdkit import RDLogger

from data import load_data, raw_to_XA
from discriminator import MolGANDiscriminator
from generator import MolGANGenerator
from utils import DEVICE, NZ, calculate_gradient_penalty, training_checks

RDLogger.DisableLog("rdApp.*")

BATCH_SIZE = 128
LAMBDA = 10
EPOCHS = 15
LR_G = 0.001
LR_D = 0.001

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
