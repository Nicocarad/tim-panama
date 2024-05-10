import comet_ml
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMCL import TIMCL
import pytorch_lightning as pl
from model_auto import LinearAutoencoder
from sklearn.model_selection import train_test_split
import torch


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder two layer 30epochs",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 1,
    "input_size": 87,
    "cutting_threshold": 0.5,
    "optimizer": "Adam"
}

comet_logger.log_hyperparams(hyper_params)


# Creazione del dataset originale
from sklearn.model_selection import train_test_split

original_dataset = TIMCL("result_df_gt_2.parquet")

indexes = range(0, len(original_dataset))


train_indexes, temp_indexes = train_test_split(
    indexes,
    train_size=0.7,
    random_state=42,
    shuffle=True,
)

val_indexes, test_indexes = train_test_split(
    temp_indexes,
    train_size=0.5,
    random_state=42,
    shuffle=True,
)

# Creazione dei subset utilizzando il dataset originale
train_dataset = Subset(original_dataset, train_indexes)
val_dataset = Subset(original_dataset, val_indexes)
test_dataset = Subset(original_dataset, test_indexes)

print("Train dataset length: ", len(train_dataset))
print("Val dataset length: ", len(val_dataset))
print("Test dataset length: ", len(test_dataset))

torch.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=False,
)



# Creazione del modello e trainer
autoencoder = LinearAutoencoder(hyper_params)
# Add CometLogger to your Trainer
trainer = pl.Trainer(max_epochs=hyper_params["epochs"], logger=comet_logger, default_root_dir="Checkpoints/")

# Allenamento del modello
trainer.fit(autoencoder, train_loader, val_dataloaders=val_loader)

# Valutazione del modello
trainer.test(autoencoder, test_loader)
