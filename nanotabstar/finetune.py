import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
from typing import Optional, Dict, Any
from .metrics import calculate_metrics

class TabSTARFinetuner:
    """
    Handles the fine-tuning process for a TabSTAR model on a specific dataset.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        accumulation_steps: int = 1,
        max_epochs: int = 10,
        patience: int = 5,
        output_dir: str = "checkpoints/finetune"
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Only optimize parameters that require gradients (LoRA + Heads)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Scheduler: OneCycleLR as specified in the paper
        # We'll need the number of steps per epoch to initialize it properly
        self.scheduler = None # Will be initialized in finetune()
        
        self.scaler = GradScaler(enabled=(device.type == 'cuda'))
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc="Training")
        for i, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                outputs = self.model(
                    feature_input_ids=batch["feature_input_ids"],
                    feature_attention_mask=batch["feature_attention_mask"],
                    feature_num_values=batch["feature_num_values"],
                    target_token_ids=batch["target_token_ids"],
                    target_attention_mask=batch["target_attention_mask"],
                    task_type=batch["task_type"]
                )
                
                if batch["task_type"] == 'classification':
                    loss = self.criterion_cls(outputs, batch["labels"])
                else:
                    # Ensure labels are [B, 1] for MSELoss
                    labels = batch["labels"].unsqueeze(1) if batch["labels"].ndim == 1 else batch["labels"]
                    loss = self.criterion_reg(outputs, labels)
                
                loss = loss / self.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                
            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({"loss": loss.item() * self.accumulation_steps})
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader, desc="Evaluating"):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        task_type = None
        
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            task_type = batch["task_type"]
            
            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                outputs = self.model(
                    feature_input_ids=batch["feature_input_ids"],
                    feature_attention_mask=batch["feature_attention_mask"],
                    feature_num_values=batch["feature_num_values"],
                    target_token_ids=batch["target_token_ids"],
                    target_attention_mask=batch["target_attention_mask"],
                    task_type=task_type
                )
                
                if task_type == 'classification':
                    loss = self.criterion_cls(outputs, batch["labels"])
                else:
                    # Ensure labels are [B, 1] for MSELoss
                    labels = batch["labels"].unsqueeze(1) if batch["labels"].ndim == 1 else batch["labels"]
                    loss = self.criterion_reg(outputs, labels)
                
                # CRITICAL: Keep the raw logits for AUC calculation
                preds = outputs
                    
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"].cpu())
            
        if not all_preds:
            return {"score": 0.0, "loss": 0.0}

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Debug: check if labels are all the same
        unique_labels = torch.unique(all_labels)
        if len(unique_labels) <= 1:
            print(f"Warning: Only one class present in {desc} set: {unique_labels.item()}")

        metrics = calculate_metrics(all_preds, all_labels, task_type)
        metrics["loss"] = total_loss / len(dataloader)
        
        return metrics

    def finetune(self, train_loader, val_loader, test_loader=None):
        best_val_metric = -float('inf')
        epochs_no_improve = 0
        
        # Initialize OneCycleLR scheduler
        total_steps = len(train_loader) * self.max_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.lr, 
            total_steps=total_steps,
            pct_start=0.1, # 10% warmup
            anneal_strategy='cos'
        )
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, desc="Validation")
            
            # Use AUC for classification, R2 for regression as primary metric
            primary_metric = val_metrics.get("auc") if val_loader.task_type == 'classification' else val_metrics.get("r2")
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Metric: {primary_metric:.4f}")
            
            if primary_metric > best_val_metric:
                best_val_metric = primary_metric
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_finetuned_model.pt"))
                print("⭐ New best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
            
        # Load best model for testing
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_finetuned_model.pt")))
        
        if test_loader:
            test_metrics = self.evaluate(test_loader, desc="Testing")
            print("\nFinal Test Results:")
            for k, v in test_metrics.items():
                print(f"{k}: {v:.4f}")
            return test_metrics
        
        return val_metrics
