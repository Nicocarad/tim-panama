import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from Autoencoders.Autoencoder.ClusterDataset import TIMCL
import pytorch_lightning as pl
from Autoencoders.Autoencoder.model.model_auto import LinearAutoencoder
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder two layer 30epochs 1917 ",
)

experiment = Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww")


hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 30,
    "input_size": 1917,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "denoise": False,
    "transofrm_type": "out-of-range",
}

comet_logger.log_hyperparams(hyper_params)


original_dataset = TIMCL(
    "result_df_gt_2_ne_type_link_1917.parquet",
    hyper_params["denoise"],
    hyper_params["transofrm_type"],
)


train_indexes = pd.read_csv("train_indexes_link.csv").values.flatten()
val_indexes = pd.read_csv("val_indexes_link.csv").values.flatten()
test_indexes = pd.read_csv("test_indexes_link.csv").values.flatten()


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


autoencoder = LinearAutoencoder(hyper_params, original_dataset.slogan)


early_stop_callback = EarlyStopping(
    monitor="val_column_wise_f1", min_delta=0.000, patience=3, verbose=False, mode="max"
)


trainer = pl.Trainer(
    max_epochs=hyper_params["epochs"],
    callbacks=[early_stop_callback],
    logger=comet_logger,
    default_root_dir="Checkpoints/",
)


trainer.fit(autoencoder, train_loader, val_dataloaders=val_loader)
trainer.test(autoencoder, test_loader)
