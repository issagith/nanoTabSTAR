# Training Process

This document describes the multi-task pretraining strategy used in **nanoTabStar**, which follows the recipe from the original TabSTAR paper while optimizing for consumer hardware.

## 1. Multi-Task Pretraining Strategy

nanoTabStar is designed as a **Foundation Model** for tabular data. To achieve this, it is pretrained on a diverse collection of datasets simultaneously.

### Global Batch Shuffling
Unlike standard training where datasets are processed sequentially, nanoTabStar uses a **Global Batch Shuffling** strategy:
1.  **Batch Generation**: For each dataset in the corpus, we generate all possible batches (up to a limit of 2048 samples per dataset per epoch).
2.  **Pure Mini-Batches**: Each batch contains samples from **only one dataset**. This ensures that the model sees a consistent set of features and target descriptions within a single forward pass.
3.  **Global Shuffle**: All batches from all datasets are pooled together and shuffled.
4.  **Task Alternation**: As the model iterates through the shuffled pool, it constantly switches between different datasets and tasks (Classification vs. Regression). This "multi-task signal" prevents the model from overfitting to a specific domain.

### Feature Consistency
To handle datasets with hundreds of columns on limited VRAM, we sample a maximum of **200 features** per dataset. These features are sampled **once per dataset at the start of training** and remain fixed throughout the epoch. This ensures that the model learns stable relationships between specific features.

---

## 2. Optimization & Scheduling

### OneCycleLR Scheduler
We use the `OneCycleLR` scheduler, which is known for "super-convergence":
- **Warmup**: The learning rate starts low and increases to `5e-5` over the first 10% of the training steps.
- **Annealing**: The learning rate then follows a cosine curve down to near zero.
- **Momentum**: The scheduler also modulates momentum (or beta1 in AdamW) in inverse proportion to the learning rate.

### Gradient Accumulation
To simulate a large global batch size (e.g., 128) while fitting on a single GPU (batch size 16), we use **Gradient Accumulation**:
- We perform 8 forward/backward passes before updating the weights.
- This stabilizes the multi-task gradients, as each weight update is informed by 8 different batches (potentially from 8 different datasets).

### Selective Unfreezing
We use a **Partial Fine-tuning** strategy for the Textual Encoder (`e5-small-v2`):
- The lower layers are kept frozen to preserve general linguistic knowledge.
- The **last 6 layers** are unfrozen to allow the model to adapt to the specific semantics of tabular feature names and values.

---

## 3. Memory Optimizations

To run a Transformer-based model on 80+ datasets with a single consumer GPU (12GB-16GB VRAM), several optimizations are active:

1.  **Gradient Checkpointing**: We trade compute for memory by not storing intermediate activations during the forward pass of the E5 backbone. They are re-calculated during the backward pass.
2.  **Chunked Textual Encoding**: Feature descriptions are processed in chunks (e.g., 512 tokens at a time) to avoid the $O(N^2)$ memory bottleneck of long sequences in the Textual Encoder.
3.  **Empty Cache**: The training loop explicitly calls `torch.cuda.empty_cache()` after validation to prevent fragmentation.

---

## 4. Evaluation and Checkpointing

- **Validation Split**: 10% of each dataset is reserved for validation.
- **Seeded Validation**: The validation loader is seeded to ensure that the "Score" is comparable across epochs.
- **Multi-Task Metric**: We track an average "Score" (AUC for classification, R² for regression).
- **Best Model**: The model state is saved to `best_model.pt` whenever the average validation score improves.

---

## 5. How to Run

### Start Pretraining
To start the pretraining loop:
```bash
python -m nanotabstar.train
```

### Monitoring
The script prints a summary at the end of each epoch:
```text
Epoch 1 Summary:
  Train Loss: 0.4521
  Val Loss:   0.4102
  Val Score:  0.7845
  New best score! Saving model...
```
