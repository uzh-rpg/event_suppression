import torch
import torch.nn as nn

class ClassificationLoss():
    def __init__(self):
        self.fp = None
        self.fn = None
        self.tp = None
        self.tn = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.F1_score = None

    @staticmethod
    def accuracy_measure(pred_binary, target):
        correct = (pred_binary == target).float().sum()  # Count correct predictions
        total = target.numel()  # Total number of elements
        return correct / total
    
    @staticmethod
    def precision_recall(tp, fp, fn):
        precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-6)
        return precision, recall
    
    @staticmethod
    def F1_score_metric(precision, recall):
        return 2 * (precision * recall) / (precision + recall + 1e-6)

    def __call__(self, pred, target):
        # Convert logits to probabilities
        pred_probs = torch.sigmoid(pred)

        # Compute predictions (threshold at 0.5)
        pred_binary = (pred_probs > 0.5).float()

        # Compute False Positives (FP) and True Positives (TP)
        self.fp = ((pred_binary == 1) & (target == 0)).float().sum()  # FP: predicted 1, but target is 0
        self.tp = ((pred_binary == 1) & (target == 1)).float().sum()  # TP: predicted 1, and target is 1
        self.fn = ((pred_binary == 0) & (target == 1)).float().sum()  # FN: predicted 0, but target is 1
        self.tn = ((pred_binary == 0) & (target == 0)).float().sum()  # TN: predicted 0, and target is 0

        self.accuracy = self.accuracy_measure(pred_binary, target)
        self.precision, self.recall = self.precision_recall(self.tp, self.fp, self.fn)
        self.F1_score = self.F1_score_metric(self.precision, self.recall)
