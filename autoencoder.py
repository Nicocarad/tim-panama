import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl

class LinearAutoencoder(pl.LightningModule):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()

        # Definizione dell'encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Definizione del decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # Sigmoid per ottenere valori tra 0 e 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('test_loss', loss)

# Preparazione dei dati
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=64, num_workers=4)
test_loader = DataLoader(mnist_val, batch_size=64, num_workers=4)

# Creazione del modello e trainer
autoencoder = LinearAutoencoder()
trainer = pl.Trainer(max_epochs=10, gpus=1)  # Modifica gpus=0 se non hai una GPU

# Allenamento del modello
trainer.fit(autoencoder, train_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)
