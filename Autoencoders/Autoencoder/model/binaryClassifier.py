import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import numpy as np


class BinaryClassifier(pl.LightningModule):
    """
    Perform binary classification on the encoded data predicting a cluster as GUASTO CAVO cluster or NON GUASTO CAVO
    Class 0: NON GUASTO CAVO
    Class 1: GUASTO CAVO

    """

    def __init__(self, encoder, input_dim, learning_rate, cutting_threshold):
        super(BinaryClassifier, self).__init__()
        self.val_outputs = []
        self.test_outputs = []
        self.result_metrics = 0

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.learning_rate = learning_rate
        self.cutting_threshold = cutting_threshold

        self.criterion = nn.BCELoss()

    def compute_loss(self, y_hat, y, scale_factor):
        loss = self.criterion(y_hat, y.float()) * scale_factor
        return loss

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.compute_loss(y_hat, y, 10)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.compute_loss(y_hat, y, 10)
        y_hat_bin = y_hat > self.cutting_threshold
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        self.val_outputs.append(
            {"y": y.cpu().numpy(), "y_hat": y_hat_bin.cpu().numpy()}
        )
        return loss

    def on_validation_epoch_end(self):

        y = np.concatenate([x["y"] for x in self.val_outputs])
        y_hat = np.concatenate([x["y_hat"] for x in self.val_outputs])
        report = classification_report(y, y_hat, output_dict=True, zero_division=0)

        # Log metrics for each class
        self.log("val_acc", report["accuracy"])
        self.log("val_precision_0", report["0"]["precision"])
        self.log("val_recall_0", report["0"]["recall"])
        self.log("val_f1_0", report["0"]["f1-score"])
        self.log("val_precision_1", report["1"]["precision"])
        self.log("val_recall_1", report["1"]["recall"])
        self.log("val_f1_1", report["1"]["f1-score"])

        # Reset for the next validation run
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.compute_loss(y_hat, y, 10)
        y_hat = y_hat > self.cutting_threshold
        self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.test_outputs.append({"y": y.cpu().numpy(), "y_hat": y_hat.cpu().numpy()})
        return loss

    def on_test_epoch_end(self):
        y = np.concatenate([x["y"] for x in self.test_outputs])
        y_hat = np.concatenate([x["y_hat"] for x in self.test_outputs])
        report = classification_report(y, y_hat, output_dict=True)

        self.log("test_acc", report["accuracy"])
        self.log("test_precision_0", report["0"]["precision"])
        self.log("test_recall_0", report["0"]["recall"])
        self.log("test_f1_0", report["0"]["f1-score"])
        self.log("test_precision_1", report["1"]["precision"])
        self.log("test_recall_1", report["1"]["recall"])
        self.log("test_f1_1", report["1"]["f1-score"])

        self.result_metrics = report

        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        return optimizer
