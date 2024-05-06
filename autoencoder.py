import comet_ml
from pytorch_lightning.loggers import CometLogger
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf


# arguments made to CometLogger are passed on to the comet_ml.Experiment class
comet_logger = CometLogger(api_key="knoxznRgLLK2INEJ9GIbmR7ww")
Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww", log_code=True)
# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 0.5,
    "steps": 100000,
    "batch_size": 50,
}

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

        # Attributo per memorizzare gli output della validazione
        self.validation_outputs = []

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        return {'x_hat': x_hat, 'target': x, 'labels': y}

    # def validation_epoch_end(self, outputs):
    #     # Memorizzazione degli output della validazione
    #     self.validation_outputs += outputs

    def on_validation_epoch_end(self):
        # Calcolo della loss della validazione
        if self.validation_outputs:
            x_hats = torch.cat([x['x_hat'] for x in self.validation_outputs])
            targets = torch.cat([x['target'] for x in self.validation_outputs])
            val_loss = nn.MSELoss()(x_hats, targets)
            self.log('val_loss', val_loss)

        # Pulizia degli output della validazione
        self.validation_outputs = []




print("INIZIO")
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
# trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.TensorBoardLogger('logs/', name='mnist_autoencoder'))  
# Add CometLogger to your Trainer
trainer = pl.Trainer(max_epochs=10, logger=comet_logger)


comet_logger.log_hyperparams(hyper_params)

# Allenamento del modello
trainer.fit(autoencoder, train_loader, val_dataloaders=test_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)

# sess.graph contains the graph definition; that enables the Graph Visualizer.

# file_writer = tf.summary.FileWriter('logs/', sess.graph)
