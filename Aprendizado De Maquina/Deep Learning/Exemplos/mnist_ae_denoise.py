"""
Exemplo de Autoencoder convolucional para remoção de ruído em MNIST.

Versão ajustada para evitar reconstruções quase pretas.

Prof. Fabiano A. Soares
"""

# -*- coding: utf-8 -*-

## ---------------------------------------------------------------------------------------- ##
# 0) Importando Bibliotecas
"""
- numpy: operações numéricas gerais.
- matplotlib: para visualização de imagens e gráficos de treino.
- torch, nn, DataLoader: núcleo do PyTorch para tensores, redes neurais e carregamento de dados.
- torchvision: fornece o dataset MNIST e transformações de imagem.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

os.makedirs("output_ae_fix", exist_ok=True)

## ---------------------------------------------------------------------------------------- ##
# 1) hiperparâmetros e device
"""
- seed: garante reprodutibilidade.
- device: escolhe GPU (cuda) se disponível, senão CPU.
- batch_size: número de exemplos por batch.
- num_epochs: número de épocas de treino.
- learning_rate: taxa de aprendizado.
- val_ratio: fração do conjunto de treino usada na validação.
- noise_factor: intensidade do ruído gaussiano.
"""
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 15
learning_rate = 1e-3
val_ratio = 0.1
noise_factor = 0.25

## ---------------------------------------------------------------------------------------- ##
# 2) Dataset MNIST, split e exemplo com ruído

transform = transforms.ToTensor()

train_full = datasets.MNIST(
    root="data_ae",
    train=True,
    download=True,
    transform=transform
)

test_set = datasets.MNIST(
    root="data_ae",
    train=False,
    download=True,
    transform=transform
)

n_val = int(len(train_full) * val_ratio)
n_train = len(train_full) - n_val
train_set, val_set = random_split(
    train_full,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(seed)
)

num_workers = 0
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def add_noise(inputs: torch.Tensor, noise_factor: float = 0.25) -> torch.Tensor:
    """Adiciona ruído gaussiano e limita a imagem ao intervalo [0, 1]."""
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    return torch.clamp(noisy, 0.0, 1.0)


example_img, example_label = train_full[0]
example_noisy = add_noise(example_img.unsqueeze(0), noise_factor=noise_factor).squeeze(0)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(example_img.squeeze(), cmap="gray", vmin=0, vmax=1)
plt.title(f"Original (classe: {example_label})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(example_noisy.squeeze(), cmap="gray", vmin=0, vmax=1)
plt.title("Com ruído")
plt.axis("off")

plt.tight_layout()
plt.savefig("output_ae_fix/exemplo_ruido.png", dpi=150)
plt.show()
plt.close()

## ---------------------------------------------------------------------------------------- ##
# 3) Modelo Autoencoder convolucional

class ConvDenoisingAutoencoder(nn.Module):
    """Autoencoder convolucional para remover ruído de imagens MNIST.

    Ajustes importantes desta versão:
    - A arquitetura foi deixada um pouco mais expressiva.
    - A perda continua sendo MSE, mas treinamos por mais épocas.
    - O ruído foi reduzido para não destruir demais o sinal original.
    """

    def __init__(self):
        super().__init__()

        # ENCODER
        # Entrada: (1, 28, 28)
        # Saída final: (32, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> (32, 7, 7)
        )

        # DECODER
        # Reconstrói gradualmente a imagem até (1, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),   # -> (8, 28, 28)
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    """
    Em PyTorch, quando criamos uma rede como uma subclasse de nn.Module,
    precisamos sobrescrever (override) o método forward(self, x).

    No forward definimos o caminho da informação pela rede:
    imagem ruidosa -> encoder -> representação comprimida -> decoder -> imagem reconstruída.
    """
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


model = ConvDenoisingAutoencoder().to(device)

## ---------------------------------------------------------------------------------------- ##
# 4) Loss, otimizador e métrica de reconstrução
"""
Usamos MSELoss porque queremos minimizar a diferença pixel a pixel
entre a imagem reconstruída e a imagem limpa.
"""
criterion = nn.MSELoss()

"""
Usamos Adam por ser uma boa escolha padrão para este tipo de arquitetura.
"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def batch_psnr(x_recon: torch.Tensor, x_clean: torch.Tensor) -> float:
    mse = torch.mean((x_recon - x_clean) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


## ---------------------------------------------------------------------------------------- ##
# 5) Training loop
train_losses, val_losses, train_psnrs, val_psnrs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    tloss, tpsnr, n_batches = 0.0, 0.0, 0

    num_batches = len(train_loader)
    print(f"\n=== Época {epoch+1}/{num_epochs} (treino AE) ===")

    for i, (x_clean, _) in enumerate(train_loader, start=1):
        x_clean = x_clean.to(device)
        x_noisy = add_noise(x_clean, noise_factor=noise_factor)

        optimizer.zero_grad()
        x_recon = model(x_noisy)
        loss = criterion(x_recon, x_clean)
        loss.backward()
        optimizer.step()

        psnr = batch_psnr(x_recon.detach(), x_clean)
        tloss += loss.item()
        tpsnr += psnr
        n_batches += 1

        if i % 10 == 0 or i == num_batches:
            frac = i / num_batches
            percent = int(frac * 100)
            print(
                f"Batch {i}/{num_batches} ({percent:3d}%) - loss {loss.item():.4f} PSNR {psnr:.2f} dB",
                end="\r"
            )

    print()
    train_losses.append(tloss / n_batches)
    train_psnrs.append(tpsnr / n_batches)

    model.eval()
    vloss, vpsnr, vn = 0.0, 0.0, 0
    with torch.inference_mode():
        for x_clean, _ in val_loader:
            x_clean = x_clean.to(device)
            x_noisy = add_noise(x_clean, noise_factor=noise_factor)
            x_recon = model(x_noisy)
            loss = criterion(x_recon, x_clean)
            vloss += loss.item()
            vpsnr += batch_psnr(x_recon, x_clean)
            vn += 1

    val_losses.append(vloss / vn)
    val_psnrs.append(vpsnr / vn)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"train loss {train_losses[-1]:.4f} PSNR {train_psnrs[-1]:.2f} dB | "
        f"val loss {val_losses[-1]:.4f} PSNR {val_psnrs[-1]:.2f} dB"
    )

# curvas de treinamento
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.xlabel("Época")
plt.ylabel("Loss (MSE)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_psnrs, label="Treino")
plt.plot(val_psnrs, label="Validação")
plt.xlabel("Época")
plt.ylabel("PSNR (dB)")
plt.legend()

plt.tight_layout()
plt.savefig("output_ae_fix/curvas_treinamento_ae.png", dpi=150)
plt.show()
plt.close()

## ---------------------------------------------------------------------------------------- ##
# 6) Teste visual: limpa, ruidosa e reconstruída

model.eval()
with torch.inference_mode():
    x_clean_batch, _ = next(iter(test_loader))
    x_clean_batch = x_clean_batch.to(device)
    x_noisy_batch = add_noise(x_clean_batch, noise_factor=noise_factor)
    x_recon_batch = model(x_noisy_batch)

x_clean_batch = x_clean_batch.cpu()
x_noisy_batch = x_noisy_batch.cpu()
x_recon_batch = x_recon_batch.cpu()

num_examples = 6
plt.figure(figsize=(9, 5))
for i in range(num_examples):
    plt.subplot(3, num_examples, i + 1)
    plt.imshow(x_clean_batch[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.title("Limpa")
    plt.axis("off")

    plt.subplot(3, num_examples, num_examples + i + 1)
    plt.imshow(x_noisy_batch[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.title("Ruidosa")
    plt.axis("off")

    plt.subplot(3, num_examples, 2 * num_examples + i + 1)
    plt.imshow(x_recon_batch[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.title("Reconstruída")
    plt.axis("off")

plt.tight_layout()
plt.savefig("output_ae_fix/exemplos_reconstrucao.png", dpi=150)
plt.show()
plt.close()
