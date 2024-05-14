import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMCL import TIMCL
import pytorch_lightning as pl
from model_auto import LinearAutoencoder

import torch
import pandas as pd
from utils_auto import split_and_save_indexes
from tqdm import tqdm


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder two layer 30epochs denoise out-of-range",
)

experiment = Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww")


# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 30,
    "input_size": 114,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "denoise": True,
    "transofrm_type": "out-of-range",
}

comet_logger.log_hyperparams(hyper_params)


original_dataset = TIMCL(
    "result_df_gt_2.parquet", hyper_params["denoise"], hyper_params["transofrm_type"]
)

indexes = range(0, len(original_dataset))


# split_and_save_indexes(indexes)


train_indexes = pd.read_csv("train_indexes.csv").values.flatten()
val_indexes = pd.read_csv("val_indexes.csv").values.flatten()
test_indexes = pd.read_csv("test_indexes.csv").values.flatten()


# Creazione dei subset utilizzando il dataset originale
train_dataset = Subset(original_dataset, train_indexes)

# original_dataset = TIMCL("result_df_gt_2.parquet") # Non applicare il denoise al validation e test
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


autoencoder = LinearAutoencoder(hyper_params, original_dataset.slogan)

trainer = pl.Trainer(
    max_epochs=hyper_params["epochs"],
    logger=comet_logger,
    default_root_dir="Checkpoints/",
)


trainer.fit(autoencoder, train_loader, val_dataloaders=val_loader)


trainer.test(autoencoder, test_loader)
