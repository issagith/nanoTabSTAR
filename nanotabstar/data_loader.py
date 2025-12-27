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
        max_length: int = 512,
        split: str = 'train',
        val_ratio: float = 0.1,
        max_features_per_batch: int = 200,
        max_samples: int = 2048,
        seed: int = 42
    ):
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.split = split
        self.val_ratio = val_ratio
        self.max_features_per_batch = max_features_per_batch
        self.max_samples = max_samples
        self.seed = seed
        
        with h5py.File(h5_path, 'r') as f:
            self.dataset_names = sorted(list(f.keys()))
            # Store indices for each dataset to respect the split
            self.dataset_indices = {}
            self.dataset_feat_indices = {} # Store fixed feature indices per dataset
            
            for ds_name in self.dataset_names:
                grp = f[ds_name]
                n_samples = grp['labels'].shape[0]
                indices = np.arange(n_samples)
                
                # Deterministic split
                rng = np.random.RandomState(seed)
                rng.shuffle(indices)
                
                split_idx = int(n_samples * (1 - val_ratio))
                if split == 'train':
                    self.dataset_indices[ds_name] = indices[:split_idx]
                else:
                    self.dataset_indices[ds_name] = indices[split_idx:]
                
                # Fixed Feature Sampling per dataset (as per paper)
                total_features = grp['feature_texts'].shape[1]
                if total_features > self.max_features_per_batch:
                    # Use the same seed for feature sampling to be consistent across splits if needed,
                    # or a different one. Here we use the provided seed.
                    feat_rng = np.random.RandomState(seed)
                    feat_indices = feat_rng.choice(total_features, self.max_features_per_batch, replace=False)
                    feat_indices.sort()
                    self.dataset_feat_indices[ds_name] = feat_indices
                else:
                    self.dataset_feat_indices[ds_name] = slice(None)
        
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
        """
        Implements the exact TabSTAR batching logic:
        1. For each dataset, sample up to 2048 examples.
        2. Create "pure" mini-batches of size 32 (one dataset per batch).
        3. Globally shuffle the list of batches to alternate between datasets.
        """
        if self.split != 'train':
            local_rng = np.random.RandomState(self.seed)
            local_random = random.RandomState(self.seed)
        else:
            # For training, we want different samples each time __iter__ is called
            local_rng = np.random.RandomState(None) 
            local_random = random.RandomState(None)

        epoch_batches = []

        with h5py.File(self.h5_path, 'r') as f:
            for ds_name in self.dataset_names:
                available_indices = self.dataset_indices[ds_name]
                n_available = len(available_indices)
                
                if n_available == 0:
                    continue

                # Select indices for this epoch (Max self.max_samples)
                if n_available > self.max_samples:
                    epoch_indices = local_rng.choice(available_indices, self.max_samples, replace=False)
                else:
                    epoch_indices = available_indices
                
                # Local shuffle to vary batches within the dataset
                epoch_indices = local_rng.permutation(epoch_indices)
                
                for i in range(0, len(epoch_indices), self.batch_size):
                    batch_indices = epoch_indices[i : i + self.batch_size]
                    if len(batch_indices) > 0: 
                        epoch_batches.append((ds_name, batch_indices))

        # Global shuffle of batches to break temporal correlation
        local_random.shuffle(epoch_batches)
        
        # Serve the data
        with h5py.File(self.h5_path, 'r') as f:
            for ds_name, batch_indices in epoch_batches:
                grp = f[ds_name]
                
                # Use fixed feature indices for this dataset
                feat_indices = self.dataset_feat_indices[ds_name]

                # H5 requires sorted indices for performance
                sorted_batch_indices = np.sort(batch_indices)
                
                raw_feat_texts = grp['feature_texts'].asstr()[sorted_batch_indices][:, feat_indices]
                feat_nums_np = grp['feature_num_values'][sorted_batch_indices][:, feat_indices]
                feat_nums = torch.from_numpy(feat_nums_np)
                
                labels_raw = grp['labels'][sorted_batch_indices]
                task_type = grp.attrs.get('task_type', 'classification')
                
                if task_type == 'classification':
                    labels = torch.from_numpy(labels_raw).long()
                else:
                    labels = torch.from_numpy(labels_raw).float()
                
                target_texts = grp['target_texts'].asstr()[:]
                
                # Tokenization
                B, M = raw_feat_texts.shape
                flat_texts = raw_feat_texts.flatten().tolist()
                encoded_features = self._tokenize_batch(flat_texts)
                
                L = encoded_features["input_ids"].shape[-1]
                feat_ids = encoded_features["input_ids"].view(B, M, L)
                feat_mask = encoded_features["attention_mask"].view(B, M, L)
                
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
                    "task_type": task_type
                }

    def __len__(self):
        # This is an approximation as the exact number of batches depends on 
        # the number of samples in each dataset (capped at self.max_samples).
        total_batches = 0
        for ds_name in self.dataset_names:
            n = min(len(self.dataset_indices[ds_name]), self.max_samples)
            total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches

class TabSTARFinetuneDataLoader:
    """
    Dataloader for fine-tuning, inference, and testing on a single dataset.
    """
    def __init__(
        self, 
        h5_path: str, 
        dataset_name: str,
        tokenizer: Any, 
        batch_size: int = 128, 
        max_length: int = 512,
        split: str = 'train',
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_features: int = 200,
        seed: int = 42,
        shuffle: bool = True
    ):
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.split = split
        self.max_features = max_features
        self.seed = seed
        self.shuffle = shuffle
        
        with h5py.File(h5_path, 'r') as f:
            if dataset_name not in f:
                raise ValueError(f"Dataset {dataset_name} not found in {h5_path}")
            
            grp = f[dataset_name]
            n_samples = grp['labels'].shape[0]
            indices = np.arange(n_samples)
            
            # Deterministic split
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            
            test_idx = int(n_samples * (1 - test_ratio))
            train_val_indices = indices[:test_idx]
            self.test_indices = indices[test_idx:]
            
            val_idx = int(len(train_val_indices) * (1 - val_ratio))
            self.train_indices = train_val_indices[:val_idx]
            self.val_indices = train_val_indices[val_idx:]
            
            if split == 'train':
                self.indices = self.train_indices
            elif split == 'val':
                self.indices = self.val_indices
            elif split == 'test':
                self.indices = self.test_indices
            else:
                self.indices = indices # Full dataset
                
            # Fixed Feature Sampling
            total_features = grp['feature_texts'].shape[1]
            if total_features > self.max_features:
                feat_rng = np.random.RandomState(seed)
                self.feat_indices = feat_rng.choice(total_features, self.max_features, replace=False)
                self.feat_indices.sort()
            else:
                self.feat_indices = slice(None)
                
            self.task_type = grp.attrs.get('task_type', 'classification')
            self.target_texts = grp['target_texts'].asstr()[:]

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle and self.split == 'train':
            np.random.shuffle(indices)
            
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.dataset_name]
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                sorted_batch_idx = np.sort(batch_idx)
                
                raw_feat_texts = grp['feature_texts'].asstr()[sorted_batch_idx][:, self.feat_indices]
                feat_nums = torch.from_numpy(grp['feature_num_values'][sorted_batch_idx][:, self.feat_indices])
                
                labels_raw = grp['labels'][sorted_batch_idx]
                
                # Map original indices back to the order of batch_idx
                # This is CRITICAL because sorted_batch_idx changes the order of labels
                # relative to the features if we don't re-align them.
                # Actually, features are also loaded using sorted_batch_idx, so they are aligned.
                # BUT, the labels must match the features.
                
                if self.task_type == 'classification':
                    labels = torch.from_numpy(labels_raw).long()
                else:
                    labels = torch.from_numpy(labels_raw).float()
                
                # Tokenization
                B, M = raw_feat_texts.shape
                flat_texts = raw_feat_texts.flatten().tolist()
                encoded_features = self._tokenize_batch(flat_texts)
                
                L = encoded_features["input_ids"].shape[-1]
                feat_ids = encoded_features["input_ids"].view(B, M, L)
                feat_mask = encoded_features["attention_mask"].view(B, M, L)
                
                encoded_targets = self._tokenize_batch(self.target_texts.tolist())
                
                yield {
                    "feature_input_ids": feat_ids,
                    "feature_attention_mask": feat_mask,
                    "feature_num_values": feat_nums,
                    "target_token_ids": encoded_targets["input_ids"],
                    "target_attention_mask": encoded_targets["attention_mask"],
                    "labels": labels,
                    "task_type": self.task_type
                }

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size