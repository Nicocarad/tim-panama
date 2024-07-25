import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ClusterDataset(Dataset):
    """Loads the cluster bitmap dataset
    Args:
        data_path (str): path to the parquet file containing the dataset
        denoise (bool): if True, the dataset is loaded with noise
        transform_type (str): if "bitflip", the dataset is loaded with bitflip noise otherwise with out-of-range noise
    """

    def __init__(self, data_path: str, denoise: bool = False, transform_type: str = None):
        self.data = pd.read_parquet(data_path)
        self.cluster_ids = self.data.index
        self.slogan = self.data.columns.tolist()
        self.denoise = denoise
        self.transform_type = transform_type
        if self.denoise:
            print("Denoising Activated")

    def __getitem__(self, idx: int):
        item = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        cluster_id = f"cluster_id2={self.cluster_ids[idx]}"
        masked_item = item.clone()

        if self.denoise:
            mask_cols = int(0.15 * len(item))
            mask_indices = np.random.choice(len(item), mask_cols, replace=False)

            if self.transform_type == "bitflip":
                masked_item[mask_indices] = 1 - masked_item[mask_indices]
            else:
                masked_item[mask_indices] = -1

        return item, masked_item, cluster_id

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    data_path = "Dataset split/Base datasets/result_df_gt_2_ne_type_link.parquet"
    dataset = ClusterDataset(data_path, denoise=True, transform_type="bitflip")
    print(dataset[2])
    print(len(dataset))
