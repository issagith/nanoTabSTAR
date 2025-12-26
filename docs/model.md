# Model Architecture

This document provides a detailed technical overview of the **nanoTabStar** architecture. nanoTabStar is a hybrid Transformer-MLP model designed to treat tabular data as a collection of semantic entities with associated numerical magnitudes.

## 1. High-Level Overview

Unlike traditional GBDT (Gradient Boosted Decision Trees) or pure MLP models, nanoTabStar leverages **Pre-trained Language Models (PLMs)** to understand the meaning of feature names and categorical values. It then fuses this semantic information with numerical values using a specialized fusion mechanism.

The architecture consists of four main stages:
1. **Semantic Encoding**: Textual descriptions are converted into dense vectors.
2. **Numerical Fusion**: Semantic vectors are combined with numerical magnitudes.
3. **Interaction Encoding**: A global Transformer models dependencies between all features.
4. **Prediction**: Task-specific heads generate the final output.

---

## 2. Input Representation

A tabular row is decomposed into $M$ features. Each feature $i$ is represented by:
- **Textual Description ($T_i$):** A string like `"Age"`, `"Job: Engineer"`, or `"City: Paris"`.
- **Numerical Value ($n_i$):** A normalized scalar (usually between 0 and 1).

For the **Target** (the value to predict), we use a special representation:
- **Target Description ($T_{target}$):** e.g., `"Predict Income"`.
- **Target Value ($n_{target}$):** Always set to **0** during the forward pass.

---

## 3. Architecture Components

### A. Textual Encoder
We use `intfloat/e5-small-v2` as the backbone, wrapped in a dedicated `TextualEncoder` class.
- **Input**: Tokenized strings for each feature and the target.
- **Pooling**: Since a feature description can be multiple tokens, we apply **Mean Pooling** (using attention masks) to obtain a single vector $e_i \in \mathbb{R}^{384}$ for each feature.
- **Flexibility**: The class includes `freeze()`, `unfreeze()`, and `unfreeze_last_k_layers(k)` methods. This allows for efficient fine-tuning by only training the top layers of the language model while keeping the lower semantic layers fixed.

### B. Numerical Components
The numerical processing is split into two distinct parts:

1. **Numerical Encoder**: A 2-layer MLP that projects the normalized scalar $n_i$ into the dense vector space $\mathbb{R}^{384}$.
2. **Numerical Fusion**: A 1-layer Transformer block that fuses the semantic vector $e_i$ and the numerical vector from the encoder.

**Fusion Process:**
1. The semantic vector and the projected scalar are treated as a sequence of 2 tokens.
2. They are passed through the fusion Transformer.
3. The output is averaged to produce a single **fused embedding** $f_i$ that contains both "what the feature is" and "what its value is".

### C. Interaction Encoder
Once we have fused embeddings $\{f_1, f_2, ..., f_M, f_{target}\}$, they are passed as a sequence into the `InteractionEncoder`.
- **Architecture**: A **6-layer Transformer Encoder**.
- **Role**: This stage allows the model to learn complex interactions (e.g., how "Age" interacts with "Income").
- **Target Contextualization**: The target embedding $f_{target}$ attends to all feature embeddings to gather the necessary information for prediction.

### D. Prediction Heads
The model maintains two shared heads (MLPs):
- **Classification Head**: Used when the target is categorical.
- **Regression Head**: Used when the target is continuous.

The output is taken from the position corresponding to the **Target token** in the sequence.

---

## 4. Memory Optimizations

To make the model "nano" (runnable on consumer GPUs), we implemented several memory-saving techniques:

### Gradient Checkpointing
We enable gradient checkpointing on the E5 backbone. This means that during the forward pass, we don't store the activations of the intermediate Transformer layers. Instead, we re-calculate them during the backward pass. This significantly reduces VRAM usage at the cost of a ~30% increase in training time.

### Chunked Textual Encoding
Since a tabular row can have many features, the total number of tokens can exceed the memory capacity of the Textual Encoder. We process the feature descriptions in **chunks** (e.g., 512 tokens at a time). This keeps the attention matrix size manageable while still allowing the model to encode all features.

---

## 5. Target-Awareness

One of the unique features of TabSTAR is that the **target is part of the input sequence**. 
- By prepending the target description (fused with $n=0$) to the features, the model "knows" what it is trying to predict before it even starts processing the features.
- This allows the same model to be used for different datasets and tasks without changing the architecture.

---

## 5. Summary of the Forward Pass

1. **Tokenize** all feature names/values and the target description.
2. **Encode** text using E5 and apply **Mean Pooling**.
3. **Project** numerical values using the Scalar MLP.
4. **Fuse** text and numbers using the `NumericalFusion` block.
5. **Concatenate** the target embedding and feature embeddings into a sequence.
6. **Process** the sequence through the **Interaction Transformer**.
7. **Extract** the target's hidden state and pass it through the appropriate **Prediction Head**.

---

## 6. Technical Specifications

| Component | Specification |
|-----------|---------------|
| Text Backbone | `intfloat/e5-small-v2` (384 dim) |
| Fusion Block | 1-layer Transformer, 2 heads |
| Interaction Encoder | 6-layer Transformer, 6 heads |
| Hidden Dimension | 384 |
| Feedforward Dim | 1536 |
| Dropout | 0.1 |
