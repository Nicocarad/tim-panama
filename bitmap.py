import pandas as pd
from tqdm import tqdm




if __name__ == "__main__":
    clusters_with_slogans = pd.read_parquet("real-time clusters/clusters_with_slogans.parquet")
    unique_slogans = clusters_with_slogans['slogan'].unique()

    # Crea una tabella con 'cluster_id' e 'cluster_id2' come indici e gli slogan come colonne
    dummies_df = pd.get_dummies(clusters_with_slogans, columns=['slogan'], prefix='', prefix_sep='')

    # Crea un oggetto tqdm per l'indicatore di avanzamento
    progress_bar = tqdm(total=len(dummies_df), desc='Processing', ncols=100)

    # Definisci una funzione che aggiorna l'indicatore di avanzamento
    def update_progress_bar(x):
        progress_bar.update()
        return x

    # Applica la funzione a ciascun gruppo e crea un nuovo dataframe con i risultati
    result_df = dummies_df.groupby(['cluster_id', 'cluster_id2']).apply(update_progress_bar).groupby(['cluster_id', 'cluster_id2']).max()

    # Chiudi l'indicatore di avanzamento
    progress_bar.close()

    result_df.to_parquet("real-time clusters/result_df.parquet")
    
 