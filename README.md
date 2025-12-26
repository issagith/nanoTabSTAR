# nanoTabStar

A compact, autonomous, and educational reimplementation of **TabSTAR**.

`nanoTabStar` is a project to reimplement the full TabSTAR pipeline—from raw data to inference—in a clean, self-contained, and didactic way. It "vends" the sophisticated preprocessing logic of the original repository into a single, readable file, making it an ideal starting point for researchers and students interested in Tabular Foundation Models (TFMs).

## Roadmap

The goal is to cover the entire lifecycle of a Tabular Foundation Model:

- [x] **Phase 1: Autonomous Preprocessing**: Replicate TabSTAR's verbalization and scaling logic without external dependencies.
- [x] **Phase 2: Corpus Generation**: Create a flexible HDF5 pretrain corpus storing raw strings for on-the-fly tokenization.
- [x] **Phase 3: Model Architecture**: Implement the hybrid Transformer + MLP architecture with Numerical Fusion.
- [x] **Phase 4: Pretraining Loop**: Implement the multi-task pretraining strategy with Gradient Accumulation and OneCycleLR.
- [ ] **Phase 5: Fine-tuning & Inference**: Tools for downstream task adaptation and evaluation.

## Key Features

- **Autonomous Preprocessing**: All logic for verbalization, numerical scaling, and date expansion is contained in `nanotabstar/preparation.py`.
- **Flexible Corpus**: Generates an HDF5 pretrain corpus storing **raw strings**. This allows experimenting with different tokenizers without regenerating the data.
- **Target-Aware Representation**: Implements the "Target-Aware" strategy where class descriptions (for classification) or target metadata (for regression) are prepended to the input features.
- **Hybrid Architecture**: 
  - **Textual Encoder**: Uses `e5-small-v2` to encode feature names and verbalized values.
  - **Numerical Fusion**: A specialized Transformer block that fuses textual embeddings with normalized numerical magnitudes.
  - **Interaction Encoder**: A global Transformer that models dependencies between all features and target tokens.
- **Multi-Task Training**: A unified training loop that handles both classification (AUC) and regression (R2) datasets simultaneously.
- **Memory Optimized**: 
  - **Gradient Checkpointing**: Recalculates activations during backward pass to save VRAM.
  - **Gradient Accumulation**: Simulates large batch sizes (e.g., 128) and ensures good meta-learning.
- **Stable Optimization**: 
  - **OneCycleLR**: Implements the 10% warmup and cosine annealing strategy from the paper.
  - **Epoch-based Batching**: Pre-calculates and shuffles "pure" mini-batches (one dataset per batch) with atmost 2048 samples per dataset.

## Repository Structure

- `nanotabstar/`: Core library.
  - `preparation.py`: The `TabSTARPreprocessor` class.
  - `model.py`: The `TabSTARModel` architecture.
  - `data_loader.py`: PyTorch-compatible dataloader with on-the-fly tokenization and train/val splits.
  - `metrics.py`: Loss and metric calculation for multi-task learning.
  - `train.py`: The main pretraining loop and execution script.
- `scripts/`:
  - `create_tabstar_corpus.py`: Script to generate the `.h5` pretrain corpus from OpenML.
- `notebooks/`:
  - `explore_corpus_generation.ipynb`: Step-by-step guide to the preprocessing logic.
  - `inference_demo.ipynb`: Manual inspection of model predictions on validation samples.
- `docs/`:
  - `pretrain_corpus.md`: Technical documentation of the HDF5 format.
  - `model.md`: Detailed explanation of the hybrid architecture.
  - `training.md`: Overview of the multi-task pretraining strategy.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Pretrain Corpus
```bash
python scripts/create_tabstar_corpus.py
```
This script downloads ~80 datasets from OpenML, processes them using the `TabSTARPreprocessor`, and saves them into a compressed HDF5 file.

### 3. Start Pretraining
To start the multi-task pretraining loop:
```bash
# Recommended: optimize CUDA memory allocation on Windows
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m nanotabstar.train
```
The script will:
1. Load the HDF5 corpus.
2. Initialize the model with `e5-small-v2`.
3. Unfreeze the last 6 layers of the textual encoder.
4. Train using OneCycleLR and Gradient Accumulation.
5. Save the best model to `data/best_model.pt`.

## Usage Example

```python
from nanotabstar.model import TabSTARModel
from nanotabstar.data_loader import TabSTARDataLoader
from transformers import AutoTokenizer

# 1. Initialize Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
model = TabSTARModel()

# 2. Load the corpus
loader = TabSTARDataLoader("data/pretrain_corpus_tabstar.h5", tokenizer=tokenizer)

for batch in loader:
    # Forward pass
    logits = model(
        feature_input_ids=batch["feature_input_ids"],
        feature_attention_mask=batch["feature_attention_mask"],
        feature_num_values=batch["feature_num_values"],
        target_token_ids=batch["target_token_ids"],
        target_attention_mask=batch["target_attention_mask"],
        task_type=batch["task_type"]
    )
    print(f"Logits shape: {logits.shape}")
    break
```

## License
This project is for educational purposes. Please refer to the original [TabSTAR](https://github.com/alanarazi7/TabSTAR) repository for the full implementation and paper.
