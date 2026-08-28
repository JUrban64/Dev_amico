import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionMIL(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.3):
        super(CrossAttentionMIL, self).__init__()
        
        # Projekce z ESM dimenzí (1280) do pracovního skrytého prostoru
        self.pocket_proj = nn.Linear(feature_dim, hidden_dim)
        self.protein_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Cross-Attention blok
        # batch_first=True => očekává tenzory ve tvaru (batch, seq, feature)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                dropout=dropout, batch_first=True)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network (FFN) podle Transformer architektury
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Klasifikátor
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, pocket_features, padding_mask, full_protein_feature):
        """
        pocket_features: [B, N, 1280] (kde N je max počet kapes v batchi)
        padding_mask: [B, N] (True znamená, že pozice je padding a má se ignorovat)
        full_protein_feature: [B, 1280]
        """
        # 1. Projekce
        B = pocket_features.size(0)
        
        # Kapsy (Keys, Values) -> [B, N, hidden_dim]
        k = self.pocket_proj(pocket_features)
        v = k
        
        # Celý protein (Query) -> [B, hidden_dim] -> [B, 1, hidden_dim]
        q = self.protein_proj(full_protein_feature).unsqueeze(1)
        
        # 2. Cross-Attention
        # Protein se "ptá" (Query) všech kapes (Key/Value)
        attn_out, attn_weights = self.cross_attn(
            query=q, 
            key=k, 
            value=v, 
            key_padding_mask=padding_mask
        )
        
        # 3. Residual connection 1 + LayerNorm 1
        out = self.norm1(attn_out.squeeze(1) + q.squeeze(1))
        
        # 4. Feed-Forward Network + Residual connection 2 + LayerNorm 2
        out = self.norm2(out + self.ffn(out))
        
        # 5. Klasifikace
        logits = self.classifier(out)
        
        return logits, attn_weights
