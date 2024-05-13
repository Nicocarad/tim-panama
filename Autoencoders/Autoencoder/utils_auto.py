import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd


def split_and_save_indexes(indexes, train_size=0.7, val_size=0.5, random_state=42):
    # Divide gli indici in set di addestramento, validazione e test
    train_indexes, temp_indexes = train_test_split(
        indexes,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    val_indexes, test_indexes = train_test_split(
        temp_indexes,
        train_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    print("Save indexes...")

    # Converti i vettori di indici in Series di pandas
    train_indexes_series = pd.Series(train_indexes)
    val_indexes_series = pd.Series(val_indexes)
    test_indexes_series = pd.Series(test_indexes)

    # Salva le Series in file CSV
    train_indexes_series.to_csv("train_indexes.csv", index=False)
    val_indexes_series.to_csv("val_indexes.csv", index=False)
    test_indexes_series.to_csv("test_indexes.csv", index=False)
