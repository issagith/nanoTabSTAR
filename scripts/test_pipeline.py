import sys
import os
import torch
from transformers import AutoTokenizer

# Ajout du dossier parent pour importer nanotabstar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanotabstar.data_loader import DatasetsDumpDataLoader

def test_loading():
    print("=== Test du DataLoader ===")
    
    # 1. Initialisation
    h5_path = 'data/pretrain_corpus.h5'
    if not os.path.exists(h5_path):
        print(f"Erreur: Le fichier {h5_path} n'existe pas. Lancez create_dump.py d'abord.")
        return

    # On demande un batch de 4 exemples pour la lisibilité
    loader = DatasetsDumpDataLoader(h5_path, batch_size=4, steps_per_epoch=1)
    
    # 2. Récupération d'un batch
    batch = next(iter(loader))
    
    print(f"\nDataset sélectionné aléatoirement : {batch['dataset_name']}")
    print("-" * 40)
    
    # 3. Inspection des dimensions (Shapes)
    # TabSTAR attend [Batch, Features, Seq_Len]
    feat_ids = batch['feature_input_ids']
    print(f"Shape des Features Tokens (B, M, L) : {feat_ids.shape}")
    print(f"Shape des Labels                    : {batch['labels'].shape}")
    print(f"Shape des Target Tokens (Classes, L) : {batch['target_token_ids'].shape}")

    # 4. Décodage pour vérification humaine (Le moment de vérité !)
    # On va re-transformer les IDs en texte pour voir ce que le modèle "lit"
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
    
    print("\n--- Visualisation du Premier Exemple du Batch ---")
    
    # On regarde la première feature de la première ligne
    first_feat_tokens = feat_ids[0, 0, :] 
    decoded_text = tokenizer.decode(first_feat_tokens, skip_special_tokens=True)
    
    # On regarde la valeur numérique associée
    num_val = batch['feature_num_values'][0, 0].item()
    
    print(f"Texte verbalisé (Feature 0) : '{decoded_text}'")
    print(f"Valeur numérique normalisée : {num_val:.4f}")
    
    print("\n--- Visualisation des Tokens Cibles (Target-Aware) ---")
    # On décode les classes possibles
    target_tokens = batch['target_token_ids']
    for i in range(target_tokens.shape[0]):
        class_text = tokenizer.decode(target_tokens[i], skip_special_tokens=True)
        print(f"Classe {i} tokenisée : '{class_text}'")

if __name__ == "__main__":
    test_loading()