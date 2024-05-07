import comet_ml
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from Autoencoders.Autoencoder import TIM_clusters
import torchvision.transforms as transforms
import pytorch_lightning as pl
from Autoencoders.Autoencoder.model import LinearAutoencoder
from sklearn.model_selection import train_test_split


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


train_dataset = TIM_clusters("result_df_gt_2.parquet")

indexes = range(0, len(train_dataset))
splitting = train_test_split(
    indexes,
    train_size=0.75,
    random_state=42,
    shuffle=True,
)
train_indexes = splitting[0]
val_indexes = splitting[1]

train_dataset = Subset(train_dataset, train_indexes)
val_dataset = Subset(train_dataset, val_indexes)

train_loader = DataLoader(
    train_dataset, batch_size=hyper_params["batch_size"], num_workers=4
)
test_loader = DataLoader(
    val_dataset, batch_size=hyper_params["batch_size"], num_workers=4
)

# Creazione del modello e trainer
autoencoder = LinearAutoencoder(hyper_params)
# Add CometLogger to your Trainer
trainer = pl.Trainer(max_epochs=hyper_params["epochs"], logger=comet_logger)

# Allenamento del modello
trainer.fit(autoencoder, train_loader, val_dataloaders=test_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)
