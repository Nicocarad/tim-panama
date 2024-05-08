import torch
import torchmetrics



class PerfectReconstruction(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # Aggiunge uno stato "perfect" per tenere traccia del numero di previsioni perfette
        self.add_state("perfect", default=torch.tensor(0), dist_reduce_fx="sum")
        # Aggiunge uno stato "total" per tenere traccia del numero totale di previsioni
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):

        assert preds.shape == target.shape
        # Aggiorna il conteggio delle previsioni perfette
        self.perfect += torch.sum(torch.all(preds == target, dim=1)).item()
        # Aggiorna il conteggio totale delle previsioni
        self.total += target.shape[0]

    def compute(self):
        # Stampa il numero di previsioni perfette e il totale delle previsioni
        print(f"Perfect: {self.perfect}")
        print(f"Total: {self.total}")
        # Calcola e restituisce l'accuracy come il rapporto tra le previsioni perfette e il totale delle previsioni
        return self.perfect / self.total
    


class BaseMetric(torchmetrics.Metric):
    def __init__(self, num_columns, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_columns = num_columns
        self.add_state(
            "true_positives", default=torch.zeros(num_columns), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_positives", default=torch.zeros(num_columns), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negatives", default=torch.zeros(num_columns), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        assert preds.shape == target.shape
        true_positives = torch.sum((preds == 1) & (target == 1), dim=0).float()
        false_positives = torch.sum((preds == 1) & (target == 0), dim=0).float()
        false_negatives = torch.sum((preds == 0) & (target == 1), dim=0).float()
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives


class ColumnWiseAccuracy(BaseMetric):
    def compute(self):
        total = self.true_positives + self.false_positives + self.false_negatives
        correct = self.true_positives
        global_accuracy = torch.mean(
            torch.where(total > 0, correct / total, torch.zeros_like(total))
        )
        return global_accuracy


class ColumnWisePrecision(BaseMetric):
    def compute(self):
        denominator = self.true_positives + self.false_positives
        global_precision = torch.mean(
            torch.where(
                denominator > 0,
                self.true_positives / denominator,
                torch.zeros_like(denominator),
            )
        )
        return global_precision


class ColumnWiseRecall(BaseMetric):
    def compute(self):
        denominator = self.true_positives + self.false_negatives
        global_recall = torch.mean(
            torch.where(
                denominator > 0,
                self.true_positives / denominator,
                torch.zeros_like(denominator),
            )
        )
        return global_recall


class ColumnWiseF1(BaseMetric):
    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        global_f1 = torch.mean(
            torch.where(torch.isfinite(f1), f1, torch.zeros_like(f1))
        )
        return global_f1
