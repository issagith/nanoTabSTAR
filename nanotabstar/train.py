import os
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Optional

from .model import TabSTARModel
from .data_loader import TabSTARDataLoader
from .metrics import calculate_loss, calculate_metrics

def run_pretraining(
    h5_path: str,
    model_name: str = "intfloat/e5-small-v2",
    batch_size: int = 16,
    lr: float = 5e-5,
    epochs: int = 10,
    steps_per_epoch: int = 100,
    val_steps: int = 20,
    unfreeze_layers: int = 6,
    device: Optional[torch.device] = None,
    save_path: str = "best_model.pt"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Starting pretraining on {device}...")
    
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
        steps_per_epoch=steps_per_epoch,
        split='train'
    )
    
    val_loader = TabSTARDataLoader(
        h5_path=h5_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        steps_per_epoch=val_steps,
        split='val'
    )
    
    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    best_val_score = -float('inf')
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            
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
            
            # Loss
            loss = calculate_loss(logits, labels, task_type)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / steps_per_epoch
        
        # 5. Evaluation Loop
        model.eval()
        val_loss = 0
        val_scores = []
        
        print(f"Running evaluation for epoch {epoch+1}...")
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
                val_scores.append(metrics["score"])
                
        avg_val_loss = val_loss / val_steps
        avg_val_score = sum(val_scores) / len(val_scores) if val_scores else 0
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Score:  {avg_val_score:.4f}")
        
        # 6. Checkpointing
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            print(f"  New best score! Saving model to {save_path}...")
            torch.save(model.state_dict(), save_path)
            
    print("Pretraining complete!")

if __name__ == "__main__":
    # Default configuration for direct execution
    H5_PATH = "data/pretrain_corpus_tabstar.h5"
    
    if not os.path.exists(H5_PATH):
        # Try to find it relative to project root if run from within nanotabstar/
        H5_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pretrain_corpus_tabstar.h5")

    if not os.path.exists(H5_PATH):
        print(f"Error: Corpus not found at {H5_PATH}. Please run scripts/create_tabstar_corpus.py first.")
    else:
        run_pretraining(
            h5_path=H5_PATH,
            epochs=10,
            steps_per_epoch=100,
            save_path="best_model.pt"
        )
