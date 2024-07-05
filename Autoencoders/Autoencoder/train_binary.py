import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from LavoriProgrammatiDataset import LavoriProgrammatiDataset
import pytorch_lightning as pl
from model.model_auto import LinearAutoencoder
import torch
import pandas as pd
from model.binaryClassifier import BinaryClassifier
import argparse


with open("config.txt", "r") as file:
    API_KEY = file.read().strip()

comet_logger = CometLogger(
    api_key=API_KEY,
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder lavori programmati",
)


experiment = Experiment(api_key=API_KEY)


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--input_size",
        type=int,
        default=32,
    )

    parse.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parse.add_argument(
        "--epochs",
        type=int,
        default=15,
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
        default=0.001,
    )

    return parse.parse_args()


args = parse_args()

hyper_params = {
    "input_size": args.input_size,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "cutting_threshold": args.cutting_threshold,
    "optimizer": args.optimizer,
    "learning_rate": args.learning_rate,
}

comet_logger.log_hyperparams(hyper_params)

hyper_params_auto = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 30,
    "input_size": 1917,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "denoise": False,
    "transofrm_type": "out-of-range",
}


original_dataset = LavoriProgrammatiDataset(
    "Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/result_df_gt_2_lavoriprogrammati_1917.parquet",
    "Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/20230101-20240101_real_time_clusters_filtered_guasto_cavo.csv",
)


train_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/train_indexes_link_lp.csv"
).values.flatten()
val_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/val_indexes_link_lp.csv"
).values.flatten()
test_indexes = pd.read_csv(
    "Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/test_indexes_link_lp.csv"
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


autoencoder = LinearAutoencoder.load_from_checkpoint(
    "Autoencoders/Autoencoder/Checkpoints/model_18epochs_1917.ckpt",
    hyper_params=hyper_params_auto,
    slogans=None,
)


encoder = autoencoder.encoder


classifier = BinaryClassifier(
    encoder,
    input_dim=hyper_params["input_size"],
    learning_rate=hyper_params["learning_rate"],
    cutting_threshold=hyper_params["cutting_threshold"],
)
trainer = pl.Trainer(
    max_epochs=hyper_params["epochs"],
    logger=comet_logger,
    default_root_dir="Checkpoints/",
)
trainer.fit(classifier, train_loader, test_loader)

trainer.test(classifier, test_loader)
