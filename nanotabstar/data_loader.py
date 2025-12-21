import h5py
import torch
import numpy as np
import random

class DatasetsDumpDataLoader:
    """
    Simulates the multi-task pretraining loop.
    Reads from the HDF5 dump. At each iteration, it randomly selects ONE dataset
    and samples a batch from it.
    """
    def __init__(self, h5_path, batch_size=32, steps_per_epoch=100):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        
        # Open file to read keys, but we'll reopen in iter to be fork-safe if needed
        with h5py.File(h5_path, 'r') as f:
            self.dataset_names = list(f.keys())
        
        print(f"Loader initialized with datasets: {self.dataset_names}")

    def __iter__(self):
        """
        Generator that yields batches indefinitely (or up to steps_per_epoch).
        """
        with h5py.File(self.h5_path, 'r') as f:
            for _ in range(self.steps_per_epoch):
                # 1. Randomly sample a dataset task 
                # "Every epoch, we randomly sample... from each dataset"
                ds_name = random.choice(self.dataset_names)
                grp = f[ds_name]
                
                n_samples = grp['labels'].shape[0]
                
                # 2. Sample indices for the batch
                if n_samples <= self.batch_size:
                    indices = np.arange(n_samples) # Take all if small dataset
                else:
                    indices = np.random.choice(n_samples, self.batch_size, replace=False)
                
                # 3. Retrieve Data
                # Note: H5 slicing must be sorted for efficiency usually, but random works ok for small batches
                indices = np.sort(indices) 
                
                # Features Text: (B, M, L)
                feat_ids = torch.from_numpy(grp['feature_input_ids'][indices])
                
                # Features Numerical: (B, M)
                feat_nums = torch.from_numpy(grp['feature_num_values'][indices])
                
                # Labels: (B,)
                labels = torch.from_numpy(grp['labels'][indices]).long()
                
                # Target Tokens: (C, L) - Constant for this batch
                # These are the "Target-Aware" tokens that define the classification task
                target_ids = torch.from_numpy(grp['target_input_ids'][:])
                
                # We need to broadcast target tokens to batch size later in the model,
                # or just pass them as metadata. Let's pass them directly.
                
                yield {
                    "feature_input_ids": feat_ids,
                    "feature_num_values": feat_nums,
                    "target_token_ids": target_ids,
                    "labels": labels,
                    "dataset_name": ds_name
                }

    def __len__(self):
        return self.steps_per_epoch