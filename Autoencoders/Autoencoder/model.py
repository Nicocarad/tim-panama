import torch.nn as nn
import torch.optim as optim
import torch
import pytorch_lightning as pl
import torchmetrics


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, hyper_params):
        super(LinearAutoencoder, self).__init__()

        self.hyper_params = hyper_params

        self.input_size = hyper_params["input_size"]
        self.cutting_threshold = hyper_params["cutting_threshold"]

        # Definizione dell'encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Definizione del decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_size),
            nn.Sigmoid(),  # Sigmoid per ottenere valori tra 0 e 1
        )

        # Inizializzazione della metrica
        self.perfect_reconstruction = PerfectReconstruction()

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
        x_hat = (x_hat > self.cutting_threshold).float()
        self.perfect_reconstruction.update(x_hat, x)
        return loss

    def on_training_epoch_end(self):
        self.log("train_perfect_reconstruction", self.perfect_reconstruction.compute())
        self.perfect_reconstruction.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hyper_params["learning_rate"])

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("test_loss", loss)
        x_hat = (x_hat > self.cutting_threshold).float()
        self.perfect_reconstruction.update(x_hat, x)

    def on_test_epoch_end(self):
        self.log("test_perfect_reconstruction", self.perfect_reconstruction.compute())
        self.perfect_reconstruction.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        # Calcolo della loss della validazione
        loss = nn.MSELoss()(x_hat, x)
        self.log("val_loss", loss)
        x_hat = (x_hat > self.cutting_threshold).float()
        self.perfect_reconstruction.update(x_hat, x)

    def on_validation_epoch_end(self):
        self.log("val_perfect_reconstruction", self.perfect_reconstruction.compute())
        self.perfect_reconstruction.reset()



class PerfectReconstruction(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("perfect", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        self.perfect += torch.sum(torch.all(preds == target, dim=1)).item()
        self.total += target.shape[0]

    def compute(self):
        return self.perfect / self.total
