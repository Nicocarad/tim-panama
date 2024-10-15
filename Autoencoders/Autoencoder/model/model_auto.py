import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model.metrics import (
    PerfectReconstruction,
    ColumnWiseAccuracy,
    ColumnWisePrecision,
    ColumnWiseRecall,
    ColumnWiseF1,
    ColumnWiseF1PerColumn,
)


class LinearAutoencoder(pl.LightningModule):
    """
    Autencoder with only MLP layers, trying to reconstruct the clusters

    """

    def __init__(self, hyper_params, slogans):
        super(LinearAutoencoder, self).__init__()

        self.hyper_params = hyper_params

        self.input_size = hyper_params["input_size"]
        self.batch_size = hyper_params["batch_size"]
        self.cutting_threshold = hyper_params["cutting_threshold"]
        self.optimizer_type = hyper_params["optimizer"]
        self.learning_rate = self.hyper_params["learning_rate"]
        self.denoise = self.hyper_params["denoise"]
        self.scale_factor = self.hyper_params["scale_factor"]

        self.slogans = slogans
        self.dropout_rate = 0.3

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.input_size),
            nn.Sigmoid(),
        )

        self.apply(self.init_weights)

        self.perfect_reconstruction = PerfectReconstruction()
        self.column_wise_accuracy = ColumnWiseAccuracy(self.input_size)
        self.column_wise_precision = ColumnWisePrecision(self.input_size)
        self.column_wise_recall = ColumnWiseRecall(self.input_size)
        self.column_wise_f1 = ColumnWiseF1(self.input_size)
        self.column_wise_f1_per_column = ColumnWiseF1PerColumn(self.input_size)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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

    def compute_loss(self, x, x_hat, scale_factor=100):
        loss = nn.BCELoss()(x_hat, x)
        scaled_loss = loss * scale_factor
        return scaled_loss

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
            x_hat = self(masked_x)
        else:
            x_hat = self(x)
        loss = self.compute_loss(x, x_hat)

        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, masked_x, _ = batch
        if self.denoise == True:
            x_hat = self(masked_x)
        else:
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
        column_wise_f1 = self.column_wise_f1_per_column.compute()
        # for i, val in enumerate(column_wise_f1):
        #     self.log(f"val_f1_{self.slogans[i]}", val, sync_dist=True)

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()
        self.column_wise_f1_per_column.reset()

    def test_step(self, batch, batch_idx):
        x, masked_x, _ = batch
        if self.denoise == True:
            x_hat = self(masked_x)
        else:
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
        column_wise_f1 = self.column_wise_f1_per_column.compute()
        # for i, val in enumerate(column_wise_f1):
        #     self.log(f"test_f1_{self.slogans[i]}", val, sync_dist=True)

        self.perfect_reconstruction.reset()
        self.column_wise_accuracy.reset()
        self.column_wise_precision.reset()
        self.column_wise_recall.reset()
        self.column_wise_f1.reset()
        self.column_wise_f1_per_column.reset()
