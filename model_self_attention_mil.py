import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionMIL(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.3):
        super(SelfAttentionMIL, self).__init__()
        
        # Projekce z ESM dimenzí (1280) do pracovního skrytého prostoru
        self.pocket_proj = nn.Linear(feature_dim, hidden_dim)
        self.protein_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Self-Attention blok
        # batch_first=True => očekává tenzory ve tvaru (batch, seq, feature)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                dropout=dropout, batch_first=True)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network (FFN) podle standardní Transformer architektury
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
        B = pocket_features.size(0)
        
        # 1. Projekce
        # Kapsy -> [B, N, hidden_dim]
        pockets = self.pocket_proj(pocket_features)
        
        # Celý protein reprezentujeme jako CLS token -> [B, 1, hidden_dim]
        cls_token = self.protein_proj(full_protein_feature).unsqueeze(1)
        
        # 2. Vytvoření sekvence spojením CLS tokenu a kapes -> [B, N+1, hidden_dim]
        x = torch.cat([cls_token, pockets], dim=1)
        
        # Aktualizace padding masky - CLS token (index 0) nikdy není padding (False)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1) # [B, N+1]
        
        # 3. Self-Attention
        attn_out, attn_weights = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=full_mask
        )
        
        # 4. Residual connection 1 + LayerNorm 1
        x = self.norm1(x + attn_out)
        
        # 5. Feed-Forward Network + Residual connection 2 + LayerNorm 2
        x = self.norm2(x + self.ffn(x))
        
        # 6. Agregace a klasifikace přes aktualizovaný CLS token
        cls_out = x[:, 0, :] # -> [B, hidden_dim]
        logits = self.classifier(cls_out)
        
        return logits, attn_weights
