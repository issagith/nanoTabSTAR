import sys
import os
# Add the project root to sys.path to allow importing nanotabstar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanotabstar import TabSTARModel, TabSTARFinetuneDataLoader, TabSTARFinetuner
from transformers import AutoTokenizer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TabSTAR on a specific dataset.")
    parser.add_argument("--h5_path", type=str, default="data/test_corpus_tabstar.h5", help="Path to H5 corpus")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to fine-tune on")
    parser.add_argument("--model_path", type=str, default="data/best_model_16.pt", help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetune", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (0.001 as per paper)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (128 as per paper)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (32 as per paper)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    
    # 1. Load Model
    model = TabSTARModel()
    if os.path.exists(args.model_path):
        print(f"Loading pre-trained weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Warning: Pre-trained model not found. Starting from scratch.")
    
    # 2. Apply LoRA
    # Alpha is fixed at 2 * r as per paper
    model = model.apply_lora(r=args.lora_r, alpha=2 * args.lora_r, dropout=args.lora_dropout)
    model.to(device)
    
    # 3. Prepare Data
    train_loader = TabSTARFinetuneDataLoader(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        split='train'
    )
    
    val_loader = TabSTARFinetuneDataLoader(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        split='val'
    )
    
    test_loader = TabSTARFinetuneDataLoader(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        split='test'
    )
    
    # 4. Fine-tune
    finetuner = TabSTARFinetuner(
        model=model,
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
        output_dir=os.path.join(args.output_dir, args.dataset_name)
    )
    
    print(f"Starting fine-tuning on {args.dataset_name}...")
    finetuner.finetune(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()
