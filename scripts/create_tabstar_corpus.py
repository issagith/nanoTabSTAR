import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_openml

# Add the project root to sys.path to allow importing nanotabstar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanotabstar.preparation import TabSTARPreprocessor

# Simplified catalog of OpenML datasets used in TabSTAR
DATASET_CATALOG = {
    "BREAST_CANCER": 15,
    "CREDIT_GERMAN": 31,
    "ADULT_INCOME": 1590,
    "CALIFORNIA_HOUSING": 43939,
    "CONCRETE_STRENGTH": 4353,
}

def prepare_dataset_locally(dataset_id, name):
    """
    Downloads from OpenML and applies TabSTAR preprocessing using the autonomous module.
    """
    print(f"  Fetching {name} (ID: {dataset_id}) from OpenML...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
    df = data.frame
    
    # Robust target column detection
    if data.target_names and len(data.target_names) > 0:
        target_col = data.target_names[0]
    elif data.target is not None:
        if isinstance(data.target, pd.Series):
            target_col = data.target.name
        elif isinstance(data.target, pd.DataFrame):
            target_col = data.target.columns[0]
        else:
            # Fallback to last column if target is just an array
            target_col = df.columns[-1]
    else:
        # Fallback to last column
        target_col = df.columns[-1]
    
    print(f"  Detected target column: {target_col}")
    
    # Determine if classification or regression
    is_cls = not pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() < 10
    
    preprocessor = TabSTARPreprocessor(is_cls=is_cls, verbose=True)
    preprocessor.fit(df.drop(columns=[target_col]), df[target_col])
    processed_data = preprocessor.transform(df.drop(columns=[target_col]), df[target_col])
    
    return {
        'x_txt': processed_data.x_txt,
        'x_num': processed_data.x_num,
        'y': processed_data.y.values,
        'target_texts': processed_data.x_txt[0, :processed_data.d_output], # First d_output columns are target tokens
        'd_output': processed_data.d_output,
        'n_features': processed_data.x_num.shape[1] - processed_data.d_output
    }

def create_corpus(output_path):
    print(f"Starting autonomous corpus creation at {output_path}")
    dt = h5py.special_dtype(vlen=str)
    
    with h5py.File(output_path, 'w') as f_out:
        for name, ds_id in tqdm(DATASET_CATALOG.items(), desc="Processing datasets"):
            try:
                data = prepare_dataset_locally(ds_id, name)
                
                ds_group = f_out.create_group(name)
                
                # Save texts
                n_samples, n_cols = data['x_txt'].shape
                d_out = data['d_output']
                
                # Feature texts (excluding target tokens)
                feature_texts = data['x_txt'][:, d_out:].astype(str)
                target_texts = data['x_txt'][0, :d_out].astype(str)
                
                ds_feat_txt = ds_group.create_dataset('feature_texts', feature_texts.shape, dtype=dt, compression='gzip')
                ds_feat_txt[:] = feature_texts
                
                ds_target_txt = ds_group.create_dataset('target_texts', target_texts.shape, dtype=dt, compression='gzip')
                ds_target_txt[:] = target_texts
                
                # Save numerical and labels
                # feature_num_values also excludes target token positions
                feature_nums = data['x_num'][:, d_out:].astype(np.float32)
                
                ds_group.create_dataset('feature_num_values', data=feature_nums, compression='gzip')
                ds_group.create_dataset('labels', data=data['y'], compression='gzip')
                
                # Metadata
                ds_group.attrs['d_output'] = d_out
                ds_group.attrs['n_features'] = data['n_features']
                ds_group.attrs['task_type'] = 'classification' if d_out > 1 else 'regression'
                
                print(f"Saved {name}")
                
            except Exception as e:
                print(f"Failed {name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    output_file = r"c:\Users\issa\dev\TFM\nanoTabStar\data\pretrain_corpus_tabstar.h5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_corpus(output_file)
