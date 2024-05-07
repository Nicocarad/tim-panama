import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch

# Verifica la disponibilità delle GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="MNIST autoencoder",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 1e-3,
    "steps": 8600,
    "batch_size": 64,
    "epochs": 20,
}

comet_logger.log_hyperparams(hyper_params)


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, hyper_params):
        super(LinearAutoencoder, self).__init__()

        self.hyper_params = hyper_params

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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hyper_params["learning_rate"])

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        # Calcolo della loss della validazione
        loss = nn.MSELoss()(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)


# Preparazione dei dati
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset = MNIST(root="./data", train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=hyper_params["batch_size"], num_workers=4)
test_loader = DataLoader(mnist_val, batch_size=hyper_params["batch_size"], num_workers=4)

# Creazione del modello e trainer
autoencoder = LinearAutoencoder(hyper_params).to(device)
# Add CometLogger to your Trainer
trainer = pl.Trainer(max_epochs=hyper_params["epochs"], logger=comet_logger)

# Allenamento del modello
trainer.fit(autoencoder, train_loader, val_dataloaders=test_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)
