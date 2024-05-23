import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMCL import TIMCL
import pytorch_lightning as pl
from model_auto import LinearAutoencoder
import torch
import pandas as pd
from model_auto import LinearAutoencoder
from binaryClassifier import BinaryClassifier


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder lavori programmati",
)

experiment = Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww")


# Configura l'autoencoder
hyper_params = {
    "input_size": 50,
    "batch_size": 32,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "denoise": False,
}


original_dataset = TIMCL("result_df_gt_2.parquet", hyper_params["denoise"], None)


train_indexes = pd.read_csv("train_indexes_lp.csv").values.flatten()
val_indexes = pd.read_csv("val_indexes_lp.csv").values.flatten()
test_indexes = pd.read_csv("test_indexes_lp.csv").values.flatten()


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
    shuffle=False,
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


autoencoder = LinearAutoencoder.load_from_checkpoint(
    "./model_30epochs.ckpt",
    hyper_params=hyper_params,
    slogans=original_dataset.slogan,
)

# Estrai l'encoder dal modello addestrato
encoder = autoencoder.encoder

# Estrai l'encoder dal modello addestrato
encoder = autoencoder.encoder

# Configura e addestra il classificatore
classifier = BinaryClassifier(encoder, input_dim=32)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(classifier, train_loader, test_loader)

# Valuta il classificatore
trainer.test(classifier, test_loader)
