import torch
import torchmetrics


class PerfectReconstruction(torchmetrics.Metric):
    """
    This metric calculates the number of perfect predictions, i.e., the number of times the entire prediction vector
    exactly matches the target vector.
    """

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
        # print(f"Perfect: {self.perfect}")
        # print(f"Total: {self.total}")
        # Calcola e restituisce l'accuracy come il rapporto tra le previsioni perfette e il totale delle previsioni
        return self.perfect / self.total


class BaseMetric(torchmetrics.Metric):
    """
    This is a base class for metrics that calculate statistics for each column of the prediction and target tensors.
    It maintains the count of true positives, false positives, and false negatives for each column.
    """

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
        self.add_state(
            "true_negatives", default=torch.zeros(num_columns), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        assert preds.shape == target.shape
        true_positives = torch.sum((preds == 1) & (target == 1), dim=0).float()
        false_positives = torch.sum((preds == 1) & (target == 0), dim=0).float()
        false_negatives = torch.sum((preds == 0) & (target == 1), dim=0).float()
        true_negatives = torch.sum((preds == 0) & (target == 0), dim=0).float()
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives
        self.true_negatives += true_negatives
        


class ColumnWiseAccuracy(BaseMetric):
    """
    This metric calculates the accuracy for each column of the prediction and target tensors.
    Accuracy is defined as the number of true positives divided by the total number of predictions.
    """

    def compute(self):
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        correct = self.true_positives + self.true_negatives
        # print("True positives: ", self.true_positives)
        # print("False positives: ", self.false_positives)
        # print("False negatives: ", self.false_negatives)
        # print("True negatives: ", self.true_negatives)
        # print(f"Total: {total}")
        # print(f"Correct: {correct}")
        
        global_accuracy = torch.mean(
            torch.where(total > 0, correct / total, torch.zeros_like(total))
        )
        return global_accuracy


class ColumnWisePrecision(BaseMetric):
    """
    This metric calculates the precision for each column of the prediction and target tensors.
    Precision is defined as the number of true positives divided by the number of true positives plus false positives.
    It shows how many of the positive predictions (columns with 1) are actually correct.
    """

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
    """
    This metric calculates the recall for each column of the prediction and target tensors.
    Recall is defined as the number of true positives divided by the number of true positives plus false negatives.
    It shows how many positives (columns with 1) are correctly identified by the model.
    """

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
    """
    This metric calculates the F1 score for each column of the prediction and target tensors.
    The F1 score is the harmonic mean of precision and recall.
    """

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        global_f1 = torch.mean(
            torch.where(torch.isfinite(f1), f1, torch.zeros_like(f1))
        )
        return global_f1
    
    
    
class ColumnWiseF1PerColumn(BaseMetric):
    """
    This metric calculates the F1 score for each individual column of the prediction and target tensors.
    The F1 score is the harmonic mean of precision and recall.
    """

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_per_column = torch.where(torch.isfinite(f1), f1, torch.zeros_like(f1))
        return f1_per_column
