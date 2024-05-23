import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

class BinaryClassifier(pl.LightningModule):
    def __init__(self, encoder, input_dim, learning_rate):
        super(BinaryClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy()
        self.precision = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1 = torchmetrics.F1()

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
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
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1(y_hat, y)
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc", self.accuracy.compute())
        self.log("test_precision", self.precision.compute())
        self.log("test_recall", self.recall.compute())
        self.log("test_f1", self.f1.compute())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        return optimizer