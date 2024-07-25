import pandas as pd
import torch


class LavoriProgrammatiDataset:
    """
    Args:
        data_path (str): path to the parquet file containing the dataset
        label_path (str): path to the csv file containing the clusters with lavori programmati labels
    """

    def __init__(self, data_path: str, label_path: str):
        self.data = pd.read_parquet(data_path)
        self.labels = pd.read_csv(label_path)
        self.labels["GUASTO CAVO"] = self.labels["GUASTO CAVO"].astype(int)
        self.data = pd.merge(
            self.data, self.labels, on="cluster_id2", how="left"
        ).set_index("cluster_id2", verify_integrity=True)
        self.cluster_ids = self.data.index

    def __getitem__(self, idx: int):
        item = torch.tensor(
            self.data.iloc[idx, :-1].dropna().values, dtype=torch.float32
        )
        label = self.data.iloc[idx, -1]
        cluster_id = f"cluster_id2={self.cluster_ids[idx]}"
        return item, label, cluster_id

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    data_path = "Dataset split/Lavoro Programmato datasets/result_df_gt_2_lavoriprogrammati.parquet"
    label_path = "Dataset split/Lavoro Programmato datasets/20230101-20240101_real_time_clusters_filtered_guasto_cavo.csv"
    dataset = LavoriProgrammatiDataset(data_path, label_path)
    print(dataset.data)
    print(dataset[0])
    print(len(dataset))
