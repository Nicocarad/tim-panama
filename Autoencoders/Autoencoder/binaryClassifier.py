
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


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
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        return optimizer

