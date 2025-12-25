import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional

class TextualEncoder(nn.Module):
    """
    Wraps a Pre-trained Language Model (PLM) to encode text into dense vectors.
    Uses mean pooling to handle variable-length sequences.
    """
    def __init__(self, model_name: str = 'intfloat/e5-small-v2'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings using the attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pooling(outputs, attention_mask)

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_k_layers(self, k: int):
        """
        Freezes the entire backbone and then unfreezes the last k layers of the encoder.
        """
        # 1. Freeze everything
        self.freeze()
        
        # 2. Unfreeze the last k layers
        # For BERT-based models like E5, layers are in backbone.encoder.layer
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer
            num_layers = len(layers)
            for i in range(max(0, num_layers - k), num_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
        else:
            # Fallback if structure is different (though E5 is standard)
            print("Warning: Could not find encoder layers to unfreeze. Check backbone structure.")

class NumericalEncoder(nn.Module):
    """
    Projects a single scalar value into a dense vector space.
    """
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (Batch, Num_Features) or (Batch, Num_Classes)
        return self.layers(x_num.unsqueeze(-1))

class NumericalFusion(nn.Module):
    """
    Fuses textual embeddings with numerical values using a small Transformer block.
    """
    def __init__(self, numerical_encoder: nn.Module, d_model: int = 384):
        super().__init__()
        self.numerical_encoder = numerical_encoder
        # Fuses the two embeddings (text and scalar)
        self.fusion_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )

    def forward(self, textual_embeddings: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # textual_embeddings: (Batch, Num_Features, D_Model)
        # x_num: (Batch, Num_Features)
        B, M, D = textual_embeddings.shape
        
        # 1. Embed numerical values
        num_embeddings = self.numerical_encoder(x_num) # (B, M, D)
        
        # 2. Prepare for fusion block (treat each feature as a sequence of 2 tokens)
        # Shape: (B * M, 2, D)
        fusion_input = torch.stack([textual_embeddings, num_embeddings], dim=2)
        fusion_input = fusion_input.view(B * M, 2, D)
        
        # 3. Apply Transformer fusion
        fused = self.fusion_block(fusion_input) # (B * M, 2, D)
        
        # 4. Average the fused tokens back to one embedding per feature
        fused_embeddings = fused.view(B, M, 2, D).mean(dim=2) # (B, M, D)
        return fused_embeddings

class InteractionEncoder(nn.Module):
    """
    A global Transformer Encoder that models dependencies between all features.
    """
    def __init__(self, d_model: int = 384, n_layers: int = 6, n_heads: int = 6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

class PredictionHead(nn.Module):
    """Simple MLP head for classification or regression."""
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TabSTARModel(nn.Module):
    """
    nanoTabStar: A hybrid Transformer-MLP architecture for Tabular Foundation Models.
    
    It combines:
    1. A Textual Encoder (E5-small) for semantic understanding.
    2. A Numerical Fusion module to integrate feature magnitudes.
    3. An Interaction Encoder (Transformer) to model dependencies between features.
    """
    def __init__(self, d_model: int = 384, n_layers: int = 6, n_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        
        # 1. Textual Encoder
        self.textual_encoder = TextualEncoder('intfloat/e5-small-v2')
        
        # 2. Numerical Components
        self.numerical_encoder = NumericalEncoder(d_model)
        self.numerical_fusion = NumericalFusion(self.numerical_encoder, d_model)
        
        # 3. Interaction Encoder
        self.interaction_encoder = InteractionEncoder(d_model, n_layers, n_heads)
        
        # 4. Prediction Heads (Shared across classes and datasets)
        self.cls_head = PredictionHead(d_model)
        self.reg_head = PredictionHead(d_model)

    def freeze_text_encoder(self):
        """Freezes the textual encoder parameters."""
        self.textual_encoder.freeze()

    def unfreeze_text_encoder(self):
        """Unfreezes the textual encoder parameters."""
        self.textual_encoder.unfreeze()

    def unfreeze_text_encoder_last_k(self, k: int = 6):
        """Unfreezes the last k layers of the textual encoder."""
        self.textual_encoder.unfreeze_last_k_layers(k)

    def forward(
        self, 
        feature_input_ids: torch.Tensor, 
        feature_attention_mask: torch.Tensor,
        feature_num_values: torch.Tensor, 
        target_token_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        task_type: str = 'classification'
    ) -> torch.Tensor:
        """
        Forward pass of the TabSTAR model.
        """
        B, M, L = feature_input_ids.shape
        C, Lt = target_token_ids.shape
        D = self.d_model
        
        # --- Step 1: Encode Target Tokens ---
        # Targets are treated as elements with numerical value n=0
        target_embeddings = self.textual_encoder(input_ids=target_token_ids, attention_mask=target_attention_mask) # (C, D)
        target_embeddings = target_embeddings.unsqueeze(0).expand(B, -1, -1) # (B, C, D)
        
        # Numerical values for targets tokens are always 0
        target_num_values = torch.zeros((B, C), device=feature_num_values.device, dtype=feature_num_values.dtype)
        
        # Fuse targets
        fused_target_embeddings = self.numerical_fusion(target_embeddings, target_num_values) # (B, C, D)
        
        # --- Step 2: Encode Feature Tokens ---
        flat_feature_ids = feature_input_ids.view(B * M, L)
        flat_feature_mask = feature_attention_mask.view(B * M, L)
        
        feature_embeddings = self.textual_encoder(input_ids=flat_feature_ids, attention_mask=flat_feature_mask) # (B*M, D)
        feature_embeddings = feature_embeddings.view(B, M, D)
        
        # Fuse features with their magnitudes
        fused_feature_embeddings = self.numerical_fusion(feature_embeddings, feature_num_values) # (B, M, D)
        
        # --- Step 3: Interaction Encoding ---
        full_sequence = torch.cat([fused_target_embeddings, fused_feature_embeddings], dim=1) # (B, C+M, D)
        encoded = self.interaction_encoder(full_sequence) # (B, C+M, D)
        
        # --- Step 4: Prediction ---
        contextualized_targets = encoded[:, :C, :] # (B, C, D)
        
        if task_type == 'regression':
            scores = self.reg_head(contextualized_targets).squeeze(-1) # (B, 1)
        else:
            scores = self.cls_head(contextualized_targets).squeeze(-1) # (B, C)
            
        return scores
