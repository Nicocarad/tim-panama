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
kf = KFold(n_splits=5)

# Configura e addestra il classificatore
classifier = BinaryClassifier(
        encoder,
        input_dim=hyper_params["input_size"],
        learning_rate=hyper_params["learning_rate"],
        cutting_threshold=hyper_params["cutting_threshold"],
        kfold=True,
    )

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

    
    trainer = pl.Trainer(
        max_epochs=hyper_params["epochs"],
        logger=comet_logger,
        default_root_dir="Checkpoints/",
    )
    trainer.fit(classifier, train_loader, test_loader)
    
results = classifier.compute_average_results()
print(results)

    
