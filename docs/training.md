# Training Process

This document describes the multi-task pretraining strategy used in **nanoTabStar**.

## 1. Multi-Task Pretraining

nanoTabStar is designed as a **Foundation Model** for tabular data. To achieve this, it is pretrained on a diverse collection of datasets simultaneously.

### Dataset Sampling
During pretraining, the model does not see one dataset at a time. Instead, for each batch:
1. A dataset is randomly sampled from the HDF5 corpus.
2. A batch of samples is drawn from that specific dataset.
3. The model performs a forward pass using the dataset's specific target descriptions.

This approach forces the model to learn a general representation of tabular structures that works across different domains (e.g., medical, financial, housing).

---

## 2. Training Objectives

The model handles two types of tasks in a single loop:

### Classification
- **Target**: Categorical labels.
- **Loss**: `CrossEntropyLoss`.
- **Metric**: `ROC-AUC` (One-vs-Rest for multi-class).
- **Mechanism**: The model produces logits for each class token provided in the input.

### Regression
- **Target**: Continuous values.
- **Loss**: `MSELoss`.
- **Metric**: `R² Score`.
- **Mechanism**: The model produces a single scalar value from the target token.

---

## 3. Optimization Strategy

### Selective Unfreezing
By default, we use a **Partial Fine-tuning** strategy for the Textual Encoder (`e5-small-v2`):
- The lower layers of the PLM are kept frozen to preserve general linguistic knowledge.
- The **last 6 layers** are unfrozen to allow the model to adapt to the specific semantics of tabular feature names and values.

### Hyperparameters
- **Optimizer**: `AdamW` with weight decay (0.01).
- **Learning Rate**: `5e-5` with a linear warmup for the first 10% of steps.
- **Gradient Clipping**: Max norm of 1.0 to ensure stability.
- **Batch Size**: 16 (default).

---

## 4. Evaluation and Checkpointing

- **Validation Split**: 10% of each dataset is reserved for validation.
- **Multi-Task Metric**: Since datasets have different scales and metrics, we track an average "Score" (AUC for classification, R² for regression).
- **Best Model**: The model state is saved to `best_model.pt` whenever the average validation score improves.

---

## 5. How to Run

### Start Pretraining
To start the pretraining loop, run the training module from the project root:
```bash
python -m nanotabstar.train
```

### Monitoring
The script uses `tqdm` to show progress for both training and evaluation phases. It prints a summary at the end of each epoch:
```text
Epoch 1 Summary:
  Train Loss: 0.4521
  Val Loss:   0.4102
  Val Score:  0.7845
  New best score! Saving model...
```
