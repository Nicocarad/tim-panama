import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TIMLP(Dataset):
    def __init__(self, data_path, label_path):

        self.data = pd.read_parquet(data_path)
        self.cluster_ids = self.data.index  # Utilizza l'indice del DataFrame
        self.labels = pd.read_csv(label_path)
        
        self.data = pd.merge(self.data,self.labels, on='cluster_id2', how='left')
        self.data = self.data.set_index('cluster_id2', verify_integrity=True)
       

    def __getitem__(self, idx):
        # Restituisce una riga del dataset e l'ID del cluster come un tensore
        item = self.data.iloc[idx, :-1].dropna().astype(float).values  # estrae tutte le colonne tranne l'ultima
        item = torch.tensor(item, dtype=torch.float32)
        label = self.data.iloc[idx, -1]  # estrae l'ultima colonna come label
        label = torch.tensor(label, dtype=torch.int64) if pd.notnull(label) else torch.tensor(0, dtype=torch.int64)
        cluster_id = f"cluster_id2={self.cluster_ids[idx]}"

        return item, label, cluster_id

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = "Dataset split/Lavoro Programmato datasets/result_df_gt_2_lavoriprogrammati.parquet"
    label_path = "Dataset split/Lavoro Programmato datasets/20230101-20240101_real_time_clusters_filtered_guasto_cavo.csv"
    dataset = TIMLP(data_path, label_path)
    print(dataset.data)
    print(dataset[0])  
    print(len(dataset))  # Stampa la lunghezza del dataset
    
