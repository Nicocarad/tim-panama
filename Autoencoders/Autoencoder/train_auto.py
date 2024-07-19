import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from ClusterDataset import ClustersDataset
import pytorch_lightning as pl
from model.model_auto import LinearAutoencoder
import torch
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
import argparse

with open("config.txt", "r") as file:
    API_KEY = file.read().strip()

comet_logger = CometLogger(
    api_key=API_KEY,
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder two layer 30epochs ",
)

experiment = Experiment(api_key=API_KEY)


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--input_size",
        type=int,
        default=1917,
    )

    parse.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parse.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parse.add_argument(
        "--cutting_threshold",
        type=float,
        default=0.5,
    )

    parse.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
    )

    parse.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parse.add_argument(
        "--denoise",
        type=bool,
        default=False,
    )

    parse.add_argument(
        "--transofrm_type",
        type=str,
        default="out-of-range",
    )

    return parse.parse_args()


args = parse_args()


hyper_params = {
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "input_size": args.input_size,
    "cutting_threshold": args.cutting_threshold,
    "optimizer": args.optimizer,
    "denoise": args.denoise,
    "transofrm_type": args.transofrm_type,
}

comet_logger.log_hyperparams(hyper_params)


original_dataset = ClustersDataset(
    "Autoencoders/Autoencoder/Dataset split/Base datasets/result_df_gt_2_ne_type_link.parquet",
    hyper_params["denoise"],
    hyper_params["transofrm_type"],
)


train_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Base datasets/train_indexes_link.csv"
).values.flatten()
val_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Base datasets/val_indexes_link.csv"
).values.flatten()
test_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Base datasets/test_indexes_link.csv"
).values.flatten()


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
