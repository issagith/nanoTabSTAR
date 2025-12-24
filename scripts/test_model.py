import os
import sys
import torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanotabstar.model import TabSTARModel
from transformers import AutoTokenizer

def test_model_forward():
    print("Testing TabSTARModel forward pass...")
    
    # 1. Setup
    d_model = 384
    batch_size = 2
    num_features = 5
    num_classes = 3
    seq_len = 16
    
    model = TabSTARModel(d_model=d_model, n_layers=2, n_heads=4)
    model.eval()
    
    # 2. Dummy Data
    feature_input_ids = torch.randint(0, 1000, (batch_size, num_features, seq_len))
    feature_attention_mask = torch.ones((batch_size, num_features, seq_len))
    feature_num_values = torch.randn(batch_size, num_features)
    target_token_ids = torch.randint(0, 1000, (num_classes, seq_len))
    target_attention_mask = torch.ones((num_classes, seq_len))
    
    # 3. Forward Pass
    with torch.no_grad():
        output = model(
            feature_input_ids=feature_input_ids,
            feature_attention_mask=feature_attention_mask,
            feature_num_values=feature_num_values,
            target_token_ids=target_token_ids,
            target_attention_mask=target_attention_mask,
            task_type='classification'
        )
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_classes)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_model_forward()
