import h5py
import torch
import numpy as np
import random
from typing import Optional, List, Dict, Any

class TabSTARDataLoader:
    """
    Simulates the multi-task pretraining loop for nanoTabStar.
    Reads from the HDF5 corpus containing raw strings and numerical values.
    Tokenizes text on-the-fly.
    """
    def __init__(
        self, 
        h5_path: str, 
        tokenizer: Any, 
        batch_size: int = 32, 
        max_length: int = 128,
        steps_per_epoch: int = 100,
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.steps_per_epoch = steps_per_epoch
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        
        with h5py.File(h5_path, 'r') as f:
            self.dataset_names = list(f.keys())
            # Store indices for each dataset to respect the split
            self.dataset_indices = {}
            for ds_name in self.dataset_names:
                n_samples = f[ds_name]['labels'].shape[0]
                indices = np.arange(n_samples)
                
                # Deterministic split
                rng = np.random.RandomState(seed)
                rng.shuffle(indices)
                
                split_idx = int(n_samples * (1 - val_ratio))
                if split == 'train':
                    self.dataset_indices[ds_name] = indices[:split_idx]
                else:
                    self.dataset_indices[ds_name] = indices[split_idx:]
        
        print(f"TabSTARDataLoader ({split}) initialized with {len(self.dataset_names)} datasets.")

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Helper to tokenize a list of strings into input_ids and attention_mask."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded

    def __iter__(self):
        with h5py.File(self.h5_path, 'r') as f:
            for _ in range(self.steps_per_epoch):
                # 1. Randomly sample a dataset
                ds_name = random.choice(self.dataset_names)
                grp = f[ds_name]
                
                available_indices = self.dataset_indices[ds_name]
                n_available = len(available_indices)
                
                if n_available == 0:
                    continue

                # 2. Sample indices from the available ones for this split
                if n_available <= self.batch_size:
                    batch_indices = available_indices
                else:
                    batch_indices = np.random.choice(available_indices, self.batch_size, replace=False)
                
                batch_indices = np.sort(batch_indices) 
                
                # 3. Retrieve Raw Data
                raw_feat_texts = grp['feature_texts'][batch_indices].astype(str)
                feat_nums = torch.from_numpy(grp['feature_num_values'][batch_indices])
                labels_raw = grp['labels'][batch_indices]
                task_type = grp.attrs.get('task_type', 'classification')
                
                if task_type == 'classification':
                    labels = torch.from_numpy(labels_raw).long()
                else:
                    labels = torch.from_numpy(labels_raw).float()
                
                # target_texts: (C,)
                target_texts = grp['target_texts'][:].astype(str)
                
                # 4. Tokenize on-the-fly
                # We flatten the (B, M) feature texts to tokenize them all at once
                B, M = raw_feat_texts.shape
                flat_texts = raw_feat_texts.flatten().tolist()
                encoded_features = self._tokenize_batch(flat_texts)
                
                # Reshape back to (B, M, L) where L is the max_length of the batch
                L = encoded_features["input_ids"].shape[-1]
                feat_ids = encoded_features["input_ids"].view(B, M, L)
                feat_mask = encoded_features["attention_mask"].view(B, M, L)
                
                # Tokenize target texts: (C, L_target)
                encoded_targets = self._tokenize_batch(target_texts.tolist())
                target_ids = encoded_targets["input_ids"]
                target_mask = encoded_targets["attention_mask"]
                
                yield {
                    "feature_input_ids": feat_ids,
                    "feature_attention_mask": feat_mask,
                    "feature_num_values": feat_nums,
                    "target_token_ids": target_ids,
                    "target_attention_mask": target_mask,
                    "labels": labels,
                    "dataset_name": ds_name,
                    "task_type": grp.attrs.get('task_type', 'unknown')
                }

    def __len__(self):
        return self.steps_per_epoch