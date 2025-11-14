import time

import torch
import torch.nn as nn

from config import DEVICE, NZ
from data import load_data, raw_to_XA
from discriminator import MolGANDiscriminator
from generator import MolGANGenerator
from utils import sample_and_check_validity
from rdkit import RDLogger

BATCH_SIZE = 128
RDLogger.DisableLog('rdApp.*')

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
    losses_G = []
    losses_D = []
    start_time = time.time()
    real_label = 0.8
    fake_label = 0.2

    loss_fn = nn.BCEWithLogitsLoss()
    optimizerG = torch.optim.SGD(params=generator.parameters(), lr=lrG)
    optimizerD = torch.optim.SGD(params=discriminator.parameters(), lr=lrD)

    for epoch in range(epochs + 1):
        counter = 0
        for X, A in data_loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            # -- Discriminator real -
            discriminator.train()
            label = torch.full(
                (len(X),), real_label, dtype=torch.float, device=DEVICE
            )
            output = discriminator.forward(A, X).squeeze(1)
            lossD_real = loss_fn(output, label)
            optimizerD.zero_grad()
            lossD_real.backward()
            D_x = output.mean().item()

            # -- Discriminator & generator --
            generator.train()
            noise = torch.randn((len(X), NZ), device=DEVICE)
            fake_A, fake_X = generator.forward(noise)
            label.fill_(fake_label)
            output = discriminator.forward(fake_A.detach(), fake_X.detach()).squeeze(1)
            lossD_fake = loss_fn(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # -- Generator --

            optimizerG.zero_grad()
            noise = torch.randn((len(X), NZ), device=DEVICE)
            fake_A, fake_X = generator.forward(noise)
            label.fill_(real_label)
            output = discriminator.forward(fake_A, fake_X).squeeze(1)
            lossG = loss_fn(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            losses_G.append(lossG.item())
            losses_D.append(lossD.item())

            if counter % (BATCH_SIZE * 500) == 0:
                print(
                    f"[{epoch}/{epochs}] (Iteration {counter}) \tLoss_D: {lossD.item():.4f}\tLoss_G: {lossG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}. ---- Dt : {time.time() - start_time:.2f}s."
                )

                # Sample molecules and check validity
                num_valid = sample_and_check_validity(generator, 100, DEVICE, NZ)
                print(f"                 Valid molecules: {num_valid}/100 ({num_valid}%)")

                start_time = time.time()
            counter += len(X)

        torch.save(generator.state_dict(), 'generator.pt')
    return losses_G, losses_D


data_loader = prepare_data()
generator = MolGANGenerator().to(DEVICE)
discriminator = MolGANDiscriminator().to(DEVICE)

losses_G, losses_D = train(data_loader, generator, discriminator, 15, 0.01, 0.0001)
