import torch
import torch.nn as nn

class AttentionMIL_ESM(nn.Module):
    """
    Attention-based Multi-Instance Learning model directly on ESM embeddings.
    """
    def __init__(self, in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3, num_heads=1, 
                 attention_temp=1.0, gated_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention_temp = attention_temp
        self.gated_attention = gated_attention
        
        mil_in_features = in_features
        
        if gated_attention:
            # Gated Attention podle Ilse et al. (2018)
            self.attention_V = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            )
            self.attention_U = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(hidden_dim, num_heads)
        else:
            # Klasická Attention
            self.attention = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_heads)  
            )
        
        # Každá hlava vygeneruje svůj vlastní embedding
        classifier_input_dim = mil_in_features * num_heads
        
        # Klasifikátor pro sloučenou reprezentaci (z báglu reziduí nebo kapes)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, padding_mask=None):
        # x shape: [B, N, in_features] or [N, in_features]
        if x.dim() == 2:
            x = x.unsqueeze(0) # [1, N, in_features]
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(0)
                
        if self.gated_attention:
            V_out = self.attention_V(x)          # [B, N, hidden_dim]
            U_out = self.attention_U(x)          # [B, N, hidden_dim]
            gated_out = V_out * U_out            # [B, N, hidden_dim]
            A_raw = self.attention_w(gated_out)  # [B, N, num_heads]
        else:
            A_raw = self.attention(x)            # [B, N, num_heads]
            
        if padding_mask is not None:
            # padding_mask shape: [B, N] (True for padding)
            A_raw = A_raw.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        
        # Aplikace teplotního škálování před softmaxem (pro zaostření/vyhlazení vah)
        A_scaled = A_raw / self.attention_temp
        
        # Softmax přes dimenzi kapes (1)
        A = torch.softmax(A_scaled, dim=1)  # [B, N, num_heads]
        
        # Seskupení instancí pro každou hlavu zvlášť
        # A.transpose(1, 2): [B, num_heads, N]
        # x: [B, N, in_features]
        Z_heads = torch.bmm(A.transpose(1, 2), x) # [B, num_heads, in_features]
        
        # Zploštění všech hlav do jednoho vektoru
        B = x.size(0)
        Z = Z_heads.view(B, -1)             # [B, num_heads * in_features]
        
        # Klasifikace celého proteinu
        logits = self.classifier(Z)         # [B, num_classes]
        return logits, A
