import torch
import torch.nn as nn
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from typing import Union, Dict, Any

def calculate_loss(predictions: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    """
    Calculates the appropriate loss based on the task type.
    """
    if task_type == 'regression':
        loss_fn = nn.MSELoss()
        # Ensure labels are float and have the same shape as predictions
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        return loss_fn(predictions, labels.float())
    else:
        loss_fn = nn.CrossEntropyLoss()
        # labels should be long for CrossEntropy
        return loss_fn(predictions, labels.long())

def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor, task_type: str) -> Dict[str, float]:
    """
    Calculates metrics (AUC for classification, R2 for regression).
    Handles cases where metrics are undefined due to single-class batches.
    """
    preds_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    if task_type == 'regression':
        # predictions are (B, 1), labels are (B,)
        preds_np = preds_np.flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = r2_score(labels_np, preds_np)
            # If all labels are the same, r2 is undefined (nan)
            if np.isnan(r2):
                r2 = 0.0
        mse = mean_squared_error(labels_np, preds_np)
        return {"r2": float(r2), "mse": float(mse), "score": float(r2)}
    else:
        # predictions are (B, C), labels are (B,)
        # For AUC, we need probabilities
        probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()
        
        num_classes = probs.shape[1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if num_classes == 2:
                    # Binary classification
                    auc = roc_auc_score(labels_np, probs[:, 1])
                else:
                    # Multi-class
                    auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='macro')
        except ValueError:
            # Fallback if only one class is present in the batch
            auc = 0.5
            
        return {"auc": float(auc), "score": float(auc)}
