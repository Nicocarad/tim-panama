import comet_ml
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMCL import TIMCL
import pytorch_lightning as pl
from model import LinearAutoencoder
from sklearn.model_selection import train_test_split
import torch


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder lr=1e-4",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 20,
    "input_size": 87,
    "cutting_threshold": 0.5,
    "optimizer": "SGD"
}

comet_logger.log_hyperparams(hyper_params)


# Creazione del dataset originale
original_dataset = TIMCL("result_df_gt_2.parquet")

indexes = range(0, len(original_dataset))
splitting = train_test_split(
    indexes,
    train_size=0.75,
    random_state=42,
    shuffle=True,
)
train_indexes = splitting[0]
val_indexes = splitting[1]

# Creazione dei subset utilizzando il dataset originale
train_dataset = Subset(original_dataset, train_indexes)
val_dataset = Subset(original_dataset, val_indexes)

print("Train dataset length: ", len(train_dataset))
print("Val dataset length: ", len(val_dataset))

torch.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=True,
)
test_loader = DataLoader(
    val_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=True,
)

print("Expected data per epoch: ", len(train_loader) * hyper_params["batch_size"])
print("Expected data per epoch: ", len(test_loader) * hyper_params["batch_size"])

# Creazione del modello e trainer
autoencoder = LinearAutoencoder(hyper_params)
# Add CometLogger to your Trainer
trainer = pl.Trainer(max_epochs=hyper_params["epochs"], logger=comet_logger)

# Allenamento del modello
trainer.fit(autoencoder, train_loader, val_dataloaders=test_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)
