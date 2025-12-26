import os
import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Optional

from .model import TabSTARModel
from .data_loader import TabSTARDataLoader
from .metrics import calculate_loss, calculate_metrics

def run_pretraining(
    h5_path: str,
    model_name: str = "intfloat/e5-small-v2",
    batch_size: int = 32,
    lr: float = 5e-5,
    epochs: int = 10,
    gradient_accumulation_steps: int = 4,
    unfreeze_layers: int = 6,
    max_samples_train: int = 2048,
    max_samples_val: int = 512,
    seed: int = 42,
    device: Optional[torch.device] = None,
    save_path: str = "best_model.pt"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Starting pretraining on {device} (seed={seed})...")
    
    # Set global seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TabSTARModel(d_model=384, n_layers=6, n_heads=6)
    
    # Apply unfreezing strategy
    print(f"Unfreezing last {unfreeze_layers} layers of the textual encoder...")
    model.unfreeze_text_encoder_last_k(unfreeze_layers)
    
    model.to(device)
    
    # 2. Initialize Dataloaders
    train_loader = TabSTARDataLoader(
        h5_path=h5_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        split='train',
        max_samples=max_samples_train,
        seed=seed
    )
    
    val_loader = TabSTARDataLoader(
        h5_path=h5_path,
        tokenizer=tokenizer,
        batch_size=batch_size * 2, # Validation can use larger batches (no gradients)
        split='val',
        max_samples=max_samples_val,
        seed=seed
    )
    
    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    
    # Total optimization steps = (total batches) / accumulation steps
    total_batches = epochs * len(train_loader)
    total_optimization_steps = total_batches // gradient_accumulation_steps
    
    # OneCycleLR as per TabSTAR paper (Section B.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr, # lr is 5e-5 by default in run_pretraining
        total_steps=total_optimization_steps,
        pct_start=0.1, # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    best_val_score = -float('inf')
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, batch in pbar:
            # Move batch to device
            feat_ids = batch["feature_input_ids"].to(device)
            feat_mask = batch["feature_attention_mask"].to(device)
            feat_nums = batch["feature_num_values"].to(device)
            target_ids = batch["target_token_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            task_type = batch["task_type"]
            
            # Forward
            logits = model(
                feature_input_ids=feat_ids,
                feature_attention_mask=feat_mask,
                feature_num_values=feat_nums,
                target_token_ids=target_ids,
                target_attention_mask=target_mask,
                task_type=task_type
            )
            
            # Loss (normalized by accumulation steps)
            loss = calculate_loss(logits, labels, task_type)
            loss = loss / gradient_accumulation_steps
            
            # Backward
            loss.backward()
            
            # Update weights every N steps
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache() # Clear cache to prevent fragmentation
            
            train_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "task": task_type[:3] # Show 'cla' or 'reg'
            })
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Evaluation Loop
        model.eval()
        val_loss = 0
        
        # Track metrics separately for classification and regression
        classif_scores = []
        reg_scores = []
        
        print(f"Running evaluation for epoch {epoch+1}...")
        torch.cuda.empty_cache() # Clear cache before evaluation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Eval]"):
                feat_ids = batch["feature_input_ids"].to(device)
                feat_mask = batch["feature_attention_mask"].to(device)
                feat_nums = batch["feature_num_values"].to(device)
                target_ids = batch["target_token_ids"].to(device)
                target_mask = batch["target_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                task_type = batch["task_type"]
                
                logits = model(
                    feature_input_ids=feat_ids,
                    feature_attention_mask=feat_mask,
                    feature_num_values=feat_nums,
                    target_token_ids=target_ids,
                    target_attention_mask=target_mask,
                    task_type=task_type
                )
                
                loss = calculate_loss(logits, labels, task_type)
                val_loss += loss.item()
                
                metrics = calculate_metrics(logits, labels, task_type)
                if task_type == 'classification':
                    classif_scores.append(metrics["score"])
                else:
                    reg_scores.append(metrics["score"])
                
                # Explicitly delete large tensors to help GC
                del feat_ids, feat_mask, feat_nums, target_ids, target_mask, logits
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate averages safely
        avg_classif = sum(classif_scores) / len(classif_scores) if classif_scores else 0.0
        avg_reg = sum(reg_scores) / len(reg_scores) if reg_scores else 0.0
        
        # Combined score for checkpointing (weighted average or simple mean)
        # We use the mean of available task types
        active_scores = []
        if classif_scores: active_scores.append(avg_classif)
        if reg_scores: active_scores.append(avg_reg)
        avg_val_score = sum(active_scores) / len(active_scores) if active_scores else 0.0
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        if classif_scores:
            print(f"  Val AUC (Classif): {avg_classif:.4f}")
        if reg_scores:
            print(f"  Val R2 (Reg):      {avg_reg:.4f}")
        print(f"  Combined Score:    {avg_val_score:.4f}")
        
        # 6. Checkpointing
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            print(f"  New best score! Saving model to {save_path}...")
            torch.save(model.state_dict(), save_path)
            
    print("Pretraining complete!")

if __name__ == "__main__":
    # Default configuration for direct execution
    H5_PATH = "data/pretrain_corpus_tabstar_16.h5"
    
    if not os.path.exists(H5_PATH):
        # Try to find it relative to project root if run from within nanotabstar/
        H5_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pretrain_corpus_tabstar.h5")

    if not os.path.exists(H5_PATH):
        print(f"Error: Corpus not found at {H5_PATH}. Please run scripts/create_tabstar_corpus.py first.")
    else:
        run_pretraining(
            h5_path=H5_PATH,
            epochs=30,
            save_path="data/best_model_64.pt"
        )
