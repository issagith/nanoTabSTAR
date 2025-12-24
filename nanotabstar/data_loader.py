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
        steps_per_epoch: int = 100
    ):
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.steps_per_epoch = steps_per_epoch
        
        with h5py.File(h5_path, 'r') as f:
            self.dataset_names = list(f.keys())
        
        print(f"TabSTARDataLoader initialized with {len(self.dataset_names)} datasets.")

    def _tokenize_batch(self, texts: List[str]) -> torch.Tensor:
        """Helper to tokenize a list of strings into input_ids."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"]

    def __iter__(self):
        with h5py.File(self.h5_path, 'r') as f:
            for _ in range(self.steps_per_epoch):
                # 1. Randomly sample a dataset
                ds_name = random.choice(self.dataset_names)
                grp = f[ds_name]
                
                n_samples = grp['labels'].shape[0]
                
                # 2. Sample indices
                if n_samples <= self.batch_size:
                    indices = np.arange(n_samples)
                else:
                    indices = np.random.choice(n_samples, self.batch_size, replace=False)
                
                indices = np.sort(indices) 
                
                # 3. Retrieve Raw Data
                # feature_texts: (B, M)
                raw_feat_texts = grp['feature_texts'][indices].astype(str)
                # feature_num_values: (B, M)
                feat_nums = torch.from_numpy(grp['feature_num_values'][indices])
                # labels: (B,)
                labels_raw = grp['labels'][indices]
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
                flat_ids = self._tokenize_batch(flat_texts)
                
                # Reshape back to (B, M, L) where L is the max_length of the batch
                L = flat_ids.shape[-1]
                feat_ids = flat_ids.view(B, M, L)
                
                # Tokenize target texts: (C, L_target)
                target_ids = self._tokenize_batch(target_texts.tolist())
                
                yield {
                    "feature_input_ids": feat_ids,
                    "feature_num_values": feat_nums,
                    "target_token_ids": target_ids,
                    "labels": labels,
                    "dataset_name": ds_name,
                    "task_type": grp.attrs.get('task_type', 'unknown')
                }

    def __len__(self):
        return self.steps_per_epoch