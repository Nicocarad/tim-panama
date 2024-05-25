import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from sklearn.metrics import classification_report
import numpy as np
import torch


class BinaryClassifier(pl.LightningModule):
    def __init__(self, encoder, input_dim, learning_rate, cutting_threshold):
        super(BinaryClassifier, self).__init__()
        self.test_outputs = []
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.learning_rate = learning_rate
        self.cutting_threshold = cutting_threshold
        self.criterion = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary", average="none")
        self.precision = torchmetrics.Precision(task="binary", average="none")
        self.recall = torchmetrics.Recall(task="binary", average="none")
        self.f1 = torchmetrics.F1Score(task="binary", average="none")
        
        self.test_outputs = []
        self.test_results = {}

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y.float())
        y_hat = y_hat > self.cutting_threshold
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1(y_hat, y)
        return loss

    def on_validation_epoch_end(self):

        self.log("val_acc", self.accuracy.compute())
        self.log("val_precision", self.precision.compute())
        self.log("val_recall", self.recall.compute())
        self.log("val_f1", self.f1.compute())

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y.float())
        y_hat = y_hat > self.cutting_threshold
        self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1(y_hat, y)
        self.test_outputs.append({"y": y.cpu().numpy(), "y_hat": y_hat.cpu().numpy()})
        return loss

    def on_test_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        y = np.concatenate([x["y"] for x in self.test_outputs])
        y_hat = np.concatenate([x["y_hat"] for x in self.test_outputs])
        report = classification_report(y, y_hat, output_dict=True)

        # Log metrics
        self.log("test_acc", acc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_acc", report["accuracy"])
        self.log("test_precision_0", report["0"]["precision"])
        self.log("test_recall_0", report["0"]["recall"])
        self.log("test_f1_0", report["0"]["f1-score"])
        self.log("test_precision_1", report["1"]["precision"])
        self.log("test_recall_1", report["1"]["recall"])
        self.log("test_f1_1", report["1"]["f1-score"])

        # Reset for the next test run
        self.test_outputs = []

        # Save metrics as an attribute
        self.test_results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        return optimizer
