# nanoTabStar (WIP)

A compact, autonomous, and educational reimplementation of **TabSTAR**.

`nanoTabStar` is an ongoing project to reimplement the full TabSTAR pipeline—from raw data to inference—in a clean, self-contained, and didactic way. It "vends" the sophisticated preprocessing logic of the original repository into a single, readable file, making it an ideal starting point for researchers and students.

## Roadmap

The goal is to cover the entire lifecycle of a Tabular Foundation Model:

- [x] **Phase 1: Autonomous Preprocessing**: Replicate TabSTAR's verbalization and scaling logic without external dependencies.
- [x] **Phase 2: Corpus Generation**: Create a flexible HDF5 pretrain corpus storing raw strings for on-the-fly tokenization.
- [ ] **Phase 3: Model Architecture**: Implement the hybrid Transformer + MLP architecture.
- [ ] **Phase 4: Pretraining Loop**: Implement the multi-task pretraining strategy.
- [ ] **Phase 5: Fine-tuning & Inference**: Tools for downstream task adaptation and evaluation.

## Key Features (Current)

- **Autonomous Preprocessing**: All logic for verbalization, numerical scaling, and date expansion is contained in `nanotabstar/preparation.py`.
- **Flexible Corpus**: Generates an HDF5 pretrain corpus storing **raw strings**. This allows for on-the-fly tokenization with any model (BERT, E5, etc.).
- **Target-Aware**: Implements the "Target-Aware" representation where class descriptions are prepended to the input.
- **Didactic**: Includes a walkthrough notebook to understand every step of the data transformation.

## Repository Structure

- `nanotabstar/`: Core library.
  - `preparation.py`: The `TabSTARPreprocessor` class.
  - `data_loader.py`: PyTorch-compatible dataloader with on-the-fly tokenization.
- `scripts/`:
  - `create_tabstar_corpus.py`: Script to generate the `.h5` pretrain corpus from OpenML datasets.
- `notebooks/`:
  - `explore_corpus_generation.ipynb`: Step-by-step guide to the preprocessing logic.
- `docs/`:
  - `pretrain_corpus.md`: Technical documentation of the HDF5 format.

## Getting Started (Current Phase)

Currently, the repository focuses on **Data Preparation**. You can generate the same pretrain corpus used by TabSTAR and explore the transformation logic.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Pretrain Corpus
```bash
python scripts/create_tabstar_corpus.py
```
This will download several datasets from OpenML and save them to `data/pretrain_corpus.h5`.

### 3. Explore the Logic
Open `notebooks/explore_corpus_generation.ipynb` to see how tabular rows are transformed into natural language.

## Usage Example

```python
from nanotabstar.preparation import TabSTARPreprocessor
from nanotabstar.data_loader import TabSTARDataLoader
from transformers import AutoTokenizer

# 1. Preprocess a custom dataset
preprocessor = TabSTARPreprocessor(is_cls=True)
# ... fit and transform ...

# 2. Load the corpus for training
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
loader = TabSTARDataLoader("data/pretrain_corpus.h5", tokenizer=tokenizer)

for batch in loader:
    print(batch["feature_input_ids"].shape) # (Batch, Num_Features, Seq_Len)
    break
```

## License
This project is for educational purposes. Please refer to the original [TabSTAR](https://github.com/Supa-Star/TabSTAR) repository for the full implementation and paper.
