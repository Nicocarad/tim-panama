import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Subset, DataLoader
from TIMLP import TIMLP
import pytorch_lightning as pl
from model_auto import LinearAutoencoder
import torch
import pandas as pd
from model_auto import LinearAutoencoder
from binaryClassifier import BinaryClassifier
from sklearn.model_selection import KFold


# Creazione del logger una sola volta
comet_logger = CometLogger(
    api_key="knoxznRgLLK2INEJ9GIbmR7ww",
    project_name="TIM_thesis",
    experiment_name="TIM autoencoder lavori programmati",
)

experiment = Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww")


# Configura l'autoencoder
hyper_params = {
    "input_size": 32,
    "batch_size": 8,
    "epochs": 30,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "learning_rate": 0.001,
}

comet_logger.log_hyperparams(hyper_params)

hyper_params_auto = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 30,
    "input_size": 113,
    "cutting_threshold": 0.5,
    "optimizer": "Adam",
    "denoise": False,
    "transofrm_type": "out-of-range",
}


test_results = {
    "accuracy": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0,
    "classification_report": {
        "0": {"precision": 0, "recall": 0, "f1-score": 0},
        "1": {"precision": 0, "recall": 0, "f1-score": 0},
    },
}


torch.manual_seed(42)


autoencoder = LinearAutoencoder.load_from_checkpoint(
    "./model_30epochs.ckpt", hyper_params=hyper_params_auto, slogans=None
)

# Estrai l'encoder dal modello addestrato
encoder = autoencoder.encoder
# Creazione del dataset
original_dataset = TIMLP(
    "result_df_gt_2_lavoriprogrammati.parquet",
    "20230101-20240101_real_time_clusters_filtered_guasto_cavo.csv",
)

# Creazione dell'oggetto KFold
kf = KFold(n_splits=2)


# Applicazione della cross-validation K-Fold
for train_indexes, test_indexes in kf.split(original_dataset):
    # Creazione dei subset utilizzando il dataset originale
    train_dataset = Subset(original_dataset, train_indexes)
    test_dataset = Subset(original_dataset, test_indexes)

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_dataset,
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

    # Configura e addestra il classificatore
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
    
    # Aggiungi i risultati del test ai risultati totali
    test_results["accuracy"] += classifier.test_results["accuracy"]
    test_results["precision"] += classifier.test_results["precision"]
    test_results["recall"] += classifier.test_results["recall"]
    test_results["f1"] += classifier.test_results["f1"]
    test_results["classification_report"]["0"]["precision"] += classifier.test_results["classification_report"]["0"]["precision"]
    test_results["classification_report"]["0"]["recall"] += classifier.test_results["classification_report"]["0"]["recall"]
    test_results["classification_report"]["0"]["f1-score"] += classifier.test_results["classification_report"]["0"]["f1-score"]
    test_results["classification_report"]["1"]["precision"] += classifier.test_results["classification_report"]["1"]["precision"]
    test_results["classification_report"]["1"]["recall"] += classifier.test_results["classification_report"]["1"]["recall"]
    test_results["classification_report"]["1"]["f1-score"] += classifier.test_results["classification_report"]["1"]["f1-score"]

# Calcola la media dei risultati
average_results = {
    "accuracy": test_results["accuracy"] / kf.get_n_splits(),
    "precision": test_results["precision"] / kf.get_n_splits(),
    "recall": test_results["recall"] / kf.get_n_splits(),
    "f1": test_results["f1"] / kf.get_n_splits(),
    "classification_report": {
        "0": {
            "precision": test_results["classification_report"]["0"]["precision"] / kf.get_n_splits(),
            "recall": test_results["classification_report"]["0"]["recall"] / kf.get_n_splits(),
            "f1-score": test_results["classification_report"]["0"]["f1-score"] / kf.get_n_splits(),
        },
        "1": {
            "precision": test_results["classification_report"]["1"]["precision"] / kf.get_n_splits(),
            "recall": test_results["classification_report"]["1"]["recall"] / kf.get_n_splits(),
            "f1-score": test_results["classification_report"]["1"]["f1-score"] / kf.get_n_splits(),
        },
    },
}

print(average_results)
