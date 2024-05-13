import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from metrics import (
    PerfectReconstruction,
    ColumnWiseAccuracy,
    ColumnWisePrecision,
    ColumnWiseRecall,
    ColumnWiseF1,
    ColumnWiseF1PerColumn,
)
import numpy as np
import pandas as pd


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, hyper_params, slogans):
        super(LinearAutoencoder, self).__init__()

        self.hyper_params = hyper_params

        self.input_size = hyper_params["input_size"]
        self.batch_size = hyper_params["batch_size"]
        self.cutting_threshold = hyper_params["cutting_threshold"]
        self.optimizer_type = hyper_params["optimizer"]
        self.learning_rate = self.hyper_params["learning_rate"]
        self.denoise = self.hyper_params["denoise"]

        self.slogans = slogans

        # Definizione dell'encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Definizione del decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_size),
            nn.Sigmoid(),
        ) 

        # Inizializzazione della metrica
        self.perfect_reconstruction = PerfectReconstruction()
        self.column_wise_accuracy = ColumnWiseAccuracy(self.input_size)
        self.column_wise_precision = ColumnWisePrecision(self.input_size)
        self.column_wise_recall = ColumnWiseRecall(self.input_size)
        self.column_wise_f1 = ColumnWiseF1(self.input_size)
        self.column_wise_f1_per_column = ColumnWiseF1PerColumn(self.input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):

        if self.optimizer_type == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adamax":
            optimizer = optim.Adamax(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        return optimizer

    def compute_loss(self, x, x_hat):
        return nn.MSELoss()(x_hat, x)

    def update_metrics(self, x, x_hat):
        x_hat = (x_hat > self.cutting_threshold).float()
        self.perfect_reconstruction.update(x_hat, x)
        self.column_wise_accuracy.update(x_hat, x)
        self.column_wise_precision.update(x_hat, x)
        self.column_wise_recall.update(x_hat, x)
        self.column_wise_f1.update(x_hat, x)
        self.column_wise_f1_per_column.update(x_hat, x)

    def training_step(self, batch, batch_idx):
        x, masked_x, _ = batch
        if self.denoise == True:
            masked_x = masked_x.view(masked_x.size(0), -1)
            x_hat = self(masked_x)
        else:
            x = x.view(x.size(0), -1)
            x_hat = self(x)
        loss = self.compute_loss(x, x_hat)

        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, masked_x, _ = batch
        if self.denoise == True:
            masked_x = masked_x.view(masked_x.size(0), -1)
            x_hat = self(masked_x)
        else:
            x = x.view(x.size(0), -1)
            x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log("val_loss", loss, sync_dist=True)
        bce = nn.BCELoss()(x_hat, x)
        self.log("val_bce", bce, sync_dist=True)
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
        column_wise_f1 = self.column_wise_f1_per_column.compute()
        for i, val in enumerate(column_wise_f1):
            self.log(f"val_f1_{self.slogans[i]}", val, sync_dist=True)

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()
        self.column_wise_f1_per_column.reset()

    def test_step(self, batch, batch_idx):
        x, masked_x, _ = batch
        if self.denoise == True:
            masked_x = masked_x.view(masked_x.size(0), -1)
            x_hat = self(masked_x)
        else:
            x = x.view(x.size(0), -1)
            x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log("test_loss", loss)
        bce = nn.BCELoss()(x_hat, x)
        self.log("val_bce", bce, sync_dist=True)
        self.update_metrics(x, x_hat)
        self.reconstructed_vectors.append(x_hat.detach().cpu().numpy())

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
        column_wise_f1 = self.column_wise_f1_per_column.compute()
        for i, val in enumerate(column_wise_f1):
            self.log(f"test_f1_{self.slogans[i]}", val, sync_dist=True)

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()
        self.column_wise_f1_per_column.reset()
