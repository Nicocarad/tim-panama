import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TIMCL(Dataset):
    def __init__(self, data_path, denoise=False, transofrm_type=None):

        self.data = pd.read_parquet(data_path)
        self.cluster_ids = self.data.index  # Utilizza l'indice del DataFrame
        self.slogan = self.data.columns.tolist()
        self.denoise = denoise
        self.transform_type = transofrm_type
        if self.denoise == True:
            print("Denoising Activated")

    def __getitem__(self, idx):
        # Restituisce una riga del dataset e l'ID del cluster come un tensore
        item = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        cluster_id = f"cluster_id2={self.cluster_ids[idx]}"
        masked_item = item.clone()

        if self.denoise == True:

            mask_cols = int(0.15 * len(item))

            mask_indices = np.random.choice(len(item), mask_cols, replace=False)

            if self.transform_type == "bitflip":
                masked_item[mask_indices] = 1 - masked_item[mask_indices]
            else:
                masked_item[mask_indices] = -1

        return item, masked_item, cluster_id

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = "Dataset split/result_df_gt_2.parquet"
    dataset = TIMCL(data_path, denoise=True, transofrm_type="bitflip")
    print(dataset.data)
    print(dataset.slogan)
    print(dataset[2])  # Stampa il tensore rappresentante la riga e l'ID del cluster
    print(len(dataset))  # Stampa la lunghezza del dataset
