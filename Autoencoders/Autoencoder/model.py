import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from metrics import PerfectReconstruction, ColumnWiseAccuracy, ColumnWisePrecision, ColumnWiseRecall, ColumnWiseF1

class LinearAutoencoder(pl.LightningModule):
    def __init__(self, hyper_params):
        super(LinearAutoencoder, self).__init__()

        self.hyper_params = hyper_params

        self.input_size = hyper_params["input_size"]
        self.batch_size = hyper_params["batch_size"]
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
        self.column_wise_accuracy = ColumnWiseAccuracy(self.input_size)
        self.column_wise_precision = ColumnWisePrecision(self.input_size)
        self.column_wise_recall = ColumnWiseRecall(self.input_size)
        self.column_wise_f1 = ColumnWiseF1(self.input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hyper_params["learning_rate"])

    def compute_loss(self, x, x_hat):
        return nn.MSELoss()(x_hat, x)

    def update_metrics(self, x, x_hat):
        x_hat = (x_hat > self.cutting_threshold).float()
        self.perfect_reconstruction.update(x_hat, x)
        self.column_wise_accuracy.update(x_hat, x)
        self.column_wise_precision.update(x_hat, x)
        self.column_wise_recall.update(x_hat, x)
        self.column_wise_f1.update(x_hat, x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log("val_loss", loss, sync_dist=True)
        self.update_metrics(x, x_hat)

    def on_validation_epoch_end(self):
        self.log(
            "val_perfect_reconstruction",
            self.perfect_reconstruction.compute(),
            sync_dist=True,
        )
        self.log(
            "val_column_wise_accuracy",
            self.column_wise_accuracy.compute(),
            sync_dist=True,
        )
        self.log(
            "val_column_wise_precision",
            self.column_wise_precision.compute(),
            sync_dist=True,
        )
        self.log(
            "val_column_wise_recall",
            self.column_wise_recall.compute(),
            sync_dist=True,
        )
        self.log(
            "val_column_wise_f1",
            self.column_wise_f1.compute(),
            sync_dist=True,
        )

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log("test_loss", loss)
        self.update_metrics(x, x_hat)

    def on_test_epoch_end(self):
        self.log(
            "test_perfect_reconstruction",
            self.perfect_reconstruction.compute(),
            sync_dist=True,
        )
        self.log(
            "test_column_wise_accuracy",
            self.column_wise_accuracy.compute(),
            sync_dist=True,
        )
        self.log(
            "test_column_wise_precision",
            self.column_wise_precision.compute(),
            sync_dist=True,
        )
        self.log(
            "test_column_wise_recall",
            self.column_wise_recall.compute(),
            sync_dist=True,
        )
        self.log(
            "test_column_wise_f1",
            self.column_wise_f1.compute(),
            sync_dist=True,
        )

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()




