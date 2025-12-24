# Pretrain Corpus Documentation

This document describes the structure and generation process of the pretrain corpus used in `nanoTabStar`. The corpus is designed to be a lightweight, educational version of the original TabSTAR pretraining data, stored in a single `.h5` file.

## Dataset Processing

The corpus is built using the core preprocessing logic from the original TabSTAR repository to ensure semantic consistency. For each dataset, the following steps are performed:

1.  **Text Cleaning**: Column names are normalized by removing special characters and standardizing whitespaces.
2.  **Date Expansion**: Temporal features are decomposed into multiple components (e.g., day of week, month, year) to help the model capture periodic patterns.
3.  **Feature Verbalization**: Every feature value is transformed into a natural language description.
    *   *Example*: A feature `Age` with value `25` becomes:
        ```
        Predictive Feature: Age
        Feature Value: 25
        ```
4.  **Numerical Scaling**: In parallel to verbalization, numerical values are standardized (Z-score) and clipped to the range `[-3, 3]`. These values are used by the MLP component of the model.
5.  **Target-Aware Tokens**: Descriptions for the target classes (classification) or the target variable (regression) are generated.
    *   *Example (Classification)*: `Target Feature: Income\nFeature Value: >50K`
    *   *Example (Regression)*: `Numerical Target Feature: Price`

## HDF5 File Structure

The corpus is stored in a single HDF5 file (`data/pretrain_corpus_tabstar.h5`). Each dataset is stored as a group named after its ID (e.g., `BIN_FINANCIAL_ADULT_INCOME`).

### Datasets within each Group

| Name | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `feature_texts` | `string` | `(N, M)` | Verbalized strings for $N$ samples and $M$ features. |
| `target_texts` | `string` | `(C,)` | Verbalized strings for $C$ target classes (or 1 for regression). |
| `feature_num_values` | `float32` | `(N, M)` | Normalized numerical values for the MLP. |
| `labels` | `int/float` | `(N,)` | Ground truth labels (class indices or regression targets). |

### Attributes

Each group contains metadata attributes:
- `task_type`: Either `'classification'` or `'regression'`.
- `d_output`: Number of output units (classes for CLS, 1 for REG).
- `n_features`: Number of predictive features.

## Usage for Training

The corpus stores **raw strings** instead of token IDs. This allows researchers to:
- Experiment with different Tokenizers (e.g., E5, BERT, RoBERTa).
- Implement "on-the-fly" tokenization in the `DataLoader`.
- Easily inspect the data being fed to the model.

To use the corpus, simply iterate through the HDF5 groups and sample batches of texts and numerical values for your training loop.
