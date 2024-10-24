import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from ClusterDataset import ClusterDataset
import pytorch_lightning as pl
from model.model_auto import LinearAutoencoder
import torch
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
import argparse

with open("config.txt", "r") as file:
    API_KEY = file.read().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cutting_threshold", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--denoise", type=str, default="false")
    parser.add_argument("--transofrm_type", type=str, default="bitflip")
    parser.add_argument("--learning_rate", type=float, default=0.0008)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--train_indexes_path", type=str)
    parser.add_argument("--val_indexes_path", type=str)
    parser.add_argument("--test_indexes_path", type=str)
    parser.add_argument("--experiment_name", type=str)
    return parser.parse_args()


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
    "scale_factor": args.scale_factor,
    "dataset_path": args.dataset_path,
    "train_indexes_path": args.train_indexes_path,
    "val_indexes_path": args.val_indexes_path,
    "test_indexes_path": args.test_indexes_path,
    "experiment_name": args.experiment_name,
}


comet_logger = CometLogger(
    api_key=API_KEY,
    project_name="Fibercop_thesis",
    experiment_name="TIM autoencoder " + hyper_params["experiment_name"],
)

experiment = Experiment(api_key=API_KEY)

comet_logger.log_hyperparams(hyper_params)


original_dataset = ClusterDataset(
    "Autoencoders/Autoencoder/Dataset split/Base datasets/"+ hyper_params["dataset_path"],
    hyper_params["denoise"],
    hyper_params["transofrm_type"],
)


train_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Base datasets/" + hyper_params["train_indexes_path"]).values.flatten()
val_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Base datasets/"+ hyper_params["val_indexes_path"]).values.flatten()
test_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Base datasets/"+ hyper_params["test_indexes_path"]).values.flatten()


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
    default_root_dir="Checkpoints/" + hyper_params["experiment_name"] + "/",
)


trainer.fit(autoencoder, train_loader, val_dataloaders=val_loader)
trainer.test(autoencoder, test_loader)
