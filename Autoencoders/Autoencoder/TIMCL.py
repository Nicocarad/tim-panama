import torch
from torch.utils.data import Dataset
import pandas as pd


class TIMCL(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path)
        self.cluster_ids = self.data.index  # Utilizza l'indice del DataFrame

    def __getitem__(self, idx):
        # Restituisce una riga del dataset e l'ID del cluster come un tensore
        item = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        cluster_id = f"cluster_id2={self.cluster_ids[idx]}"

        return item, cluster_id

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = "result_df_gt_2.parquet"
    dataset = TIMCL(data_path)
    print(dataset[2])  # Stampa il tensore rappresentante la riga e l'ID del cluster
    print(len(dataset))  # Stampa la lunghezza del dataset
