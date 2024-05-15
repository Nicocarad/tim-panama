import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMCL import TIMCL
import pytorch_lightning as pl
from model_auto import LinearAutoencoder

import pandas as pd


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder evaluate test ",
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
    "denoise": False,
    "transofrm_type": "bitflip",
}

original_dataset = TIMCL("result_df_gt_2.parquet", False, None)

autoencoder = LinearAutoencoder.load_from_checkpoint(
    "./epoch=29-step=283530.ckpt",
    hyper_params=hyper_params,
    slogans=original_dataset.slogan,
)


test_indexes = pd.read_csv("test_indexes.csv").values.flatten()


test_dataset = Subset(original_dataset, test_indexes)


print("Test dataset length: ", len(test_dataset))


test_loader = DataLoader(
    test_dataset,
    batch_size=hyper_params["batch_size"],
    num_workers=1,
    drop_last=True,
    shuffle=False,
)

# Crea un Trainer
trainer = pl.Trainer(
    max_epochs=hyper_params["epochs"],
    logger=comet_logger,
    default_root_dir="Checkpoints/",
)

# Esegui il test
trainer.test(autoencoder, test_loader)
