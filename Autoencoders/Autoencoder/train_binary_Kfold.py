import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from LavoriProgrammatiDataset import LavoriProgrammatiDataset
import pytorch_lightning as pl
from model.model_auto import LinearAutoencoder
import torch
import pandas as pd
from model.model_auto import LinearAutoencoder
from model.binaryClassifier import BinaryClassifier
import numpy as np
import argparse

NUM_FOLD = 5
results = []

with open("config.txt", "r") as file:
    API_KEY = file.read().strip()

comet_logger = CometLogger(
    api_key=API_KEY,
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder lavori programmati Kfold",
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
        default=64,
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


train_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/train_indexes_link_lp.csv").values.flatten()
val_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/val_indexes_link_lp.csv").values.flatten()
test_indexes = pd.read_csv("Autoencoders/Autoencoder/Dataset split/Lavoro Programmato datasets/test_indexes_link_lp.csv").values.flatten()


all_indexes = np.concatenate([train_indexes, val_indexes, test_indexes])


fold_size = len(all_indexes) // NUM_FOLD

fold1 = all_indexes[0:fold_size]
fold2 = all_indexes[fold_size : 2 * fold_size]
fold3 = all_indexes[2 * fold_size : 3 * fold_size]
fold4 = all_indexes[3 * fold_size : 4 * fold_size]
fold5 = all_indexes[4 * fold_size :]


torch.manual_seed(42)


autoencoder = LinearAutoencoder.load_from_checkpoint(
    "Autoencoders/Autoencoder/Checkpoints/model_18epochs_1917.ckpt", hyper_params=hyper_params_auto, slogans=None
)

encoder = autoencoder.encoder


for iter in range(NUM_FOLD):

    folds = [fold1, fold2, fold3, fold4, fold5]
    train_idx = np.concatenate([f for i, f in enumerate(folds) if i != iter])
    val_idx = folds[iter]

    train_dataset = Subset(original_dataset, train_idx)
    val_dataset = Subset(original_dataset, val_idx)

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

    trainer.fit(classifier, train_loader)

    trainer.test(classifier, val_loader)

    results.append(classifier.result_metrics)

print(results)


sum_metrics = {
    "0": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
    "1": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
    "accuracy": 0,
}


for report in results:

    sum_metrics["0"]["precision"] += report["0"]["precision"]
    sum_metrics["0"]["recall"] += report["0"]["recall"]
    sum_metrics["0"]["f1-score"] += report["0"]["f1-score"]
    sum_metrics["0"]["support"] += report["0"]["support"]

    sum_metrics["1"]["precision"] += report["1"]["precision"]
    sum_metrics["1"]["recall"] += report["1"]["recall"]
    sum_metrics["1"]["f1-score"] += report["1"]["f1-score"]
    sum_metrics["1"]["support"] += report["1"]["support"]

    sum_metrics["accuracy"] += report["accuracy"]


avg_metrics = {
    "0": {metric: total / NUM_FOLD for metric, total in sum_metrics["0"].items()},
    "1": {metric: total / NUM_FOLD for metric, total in sum_metrics["1"].items()},
    "accuracy": sum_metrics["accuracy"] / NUM_FOLD,
}


comet_logger.experiment.log_metrics(avg_metrics["0"], prefix="avg/0/")
comet_logger.experiment.log_metrics(avg_metrics["1"], prefix="avg/1/")
comet_logger.experiment.log_metric("avg_accuracy", avg_metrics["accuracy"])


print("Average results", avg_metrics)
