import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd





def create_df(dataset, indexes):
    data = []
    for i in indexes:
        features = dataset[i][0].numpy()  # Converti il tensore in un array numpy
        cluster_id = dataset[i][1]
        row = [cluster_id] + list(
            features
        )  # Crea una lista con l'ID del cluster e le caratteristiche
        data.append(row)
    # Crea un DataFrame con le colonne corrispondenti all'ID del cluster e alle caratteristiche
    df = pd.DataFrame(
        data, columns=["cluster_id"] + [f"{feature}" for feature in dataset.slogan]
    )
    df.set_index(
        "cluster_id", inplace=True
    )  # Imposta 'cluster_id' come indice del DataFrame
    return df
