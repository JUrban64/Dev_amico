import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class GatedAttentionPool(nn.Module):
    """
    Gated Attention Pooling pro agregaci reziduí uvnitř jedné kapsy.
    Umožňuje modelu naučit se, která rezidua v kapse jsou klíčová pro vazbu.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn_v = nn.Linear(dim, dim // 2)
        self.attn_u = nn.Linear(dim, dim // 2)
        self.attn_w = nn.Linear(dim // 2, 1)

    def forward(self, h, batch_idx):
        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        scores = self.attn_w(v * u).squeeze(-1) # [N]
        
        from torch_geometric.utils import softmax
        weights = softmax(scores, batch_idx) # [N]
        out = global_mean_pool(h * weights.unsqueeze(-1) * batch_idx.bincount()[batch_idx].unsqueeze(-1), batch_idx)
        return out


class EquivariantPocketEncoder(nn.Module):
    """
    3D E(3)-Equivariant Graph Neural Network pro extrakci geometrických reprezentací kapes.
    Zpracovává uzly (rezidua s ESM-2 embeddingy) a jejich 3D souřadnice (C-alpha) s hranami podle vzdálenosti.
    Aktualizuje jak souřadnice, tak vlastnosti uzlů a agreguje je do jednoho embeddingu kapsy.
    """
    def __init__(self, node_dim=1280, hidden_dim=128, num_gnn_layers=2, dropout=0.3, coord_noise=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_noise = coord_noise
        
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        try:
            from egnn_pytorch import EGNN_Sparse
            self.gnn_layers = nn.ModuleList([
                EGNN_Sparse(
                    feats_dim=hidden_dim,
                    pos_dim=3,
                    m_dim=hidden_dim,
                    update_feats=True,
                    update_coors=True,
                    dropout=dropout,
                    norm_feats=True,
                    norm_coors=True
                )
                for _ in range(num_gnn_layers)
            ])
            self.has_egnn = True
        except ImportError:
            print("Upozornění: egnn_pytorch není k dispozici, používám GNN fallback.")
            self.has_egnn = False
            self.fallback_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
            ])
            
        self.gated_pool = GatedAttentionPool(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos, edge_index, batch_idx=None):
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # 3D Data Augmentation: Coordinate Jittering během tréninku
        if self.training and self.coord_noise > 0:
            pos = pos + torch.randn_like(pos) * self.coord_noise
            
        feats = self.node_proj(x)
        
        if self.has_egnn:
            # Centrování souřadnic vzhledem k těžišti každé kapsy (SE(3) translace)
            mean_coors = global_mean_pool(pos, batch_idx)
            coors = pos - mean_coors[batch_idx]
            
            x_in = torch.cat([coors, feats], dim=-1)
            for layer in self.gnn_layers:
                x_in = layer(x=x_in, edge_index=edge_index, batch=batch_idx)
                
            coors = x_in[:, :3]
            feats = x_in[:, 3:]
        else:
            for layer in self.fallback_layers:
                feats = F.gelu(layer(feats))
                
        # Dvojitý pooling: Gated Attention + Global Mean Pool
        gated_emb = self.gated_pool(feats, batch_idx)
        mean_emb = global_mean_pool(feats, batch_idx)
        
        pocket_emb = self.out_proj(torch.cat([gated_emb, mean_emb], dim=-1))
        return pocket_emb


class TransformerSelfAttentionLayer(nn.Module):
    """
    Jeden blok Self-Attention Transformeru s uchováním pozornostních vah (attention weights)
    pro následnou interpretaci a vizualizaci důležitosti jednotlivých kapes.
    """
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, key_padding_mask=None):
        """
        x: [B, Seq_len, hidden_dim]
        key_padding_mask: [B, Seq_len] (True pro padding pozice)
        """
        attn_out, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class EGNN_Self_Attention_MIL(nn.Module):
    """
    Architektura kombinující 3D EGNN reprezentaci kapes s globálním sekvenčním embeddingem
    pomocí plného Self-Attention mechanismu (Transformer).
    
    1. Kapsy jsou modelovány jako 3D reziduální grafy a kódovány pomocí EGNN (Equivariant GNN),
       který respektuje SE(3) symetrie a extrahuje geometricko-chemický embedding pro každou kapsu.
    2. Protein má svůj standardní sekvenční embedding (např. ESM-2 [1280]), který je promítnut
       do dimenze modelu a slouží jako globální [CLS] token sekvence.
    3. Spojení do sekvence tokenů: [CLS_protein, Pocket_1, Pocket_2, ..., Pocket_K].
    4. Multi-Head Self-Attention Transformer:
       - Proteinový [CLS] token se dívá na všechny kapsy (kombinuje globální kontext s lokálními vazebnými místy).
       - Kapsy komunikují mezi sebou (multi-pocket kooperativita, alosterické interakce, složitá vazebná místa).
       - Kapsy se dívají na globální protein (podmínění kapsy rodinou/sekvencí proteinu).
    5. Klasifikační hlava: Z aktualizovaného [CLS] tokenu (obohaceného o informace ze všech kapes)
       a agregovaných kapes se predikují logity pro 5 kofaktorových tříd.
    """
    def __init__(self,
                 node_dim=1280,
                 full_protein_dim=1280,
                 hidden_dim=128,
                 num_gnn_layers=2,
                 num_heads=4,
                 num_attn_layers=1,
                 num_classes=5,
                 dropout=0.3,
                 pocket_drop_prob=0.1,
                 use_pocket_pooling_in_head=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pocket_drop_prob = pocket_drop_prob
        self.use_pocket_pooling_in_head = use_pocket_pooling_in_head
        
        # 1. EGNN Kodér 3D kapesních grafů
        self.pocket_encoder = EquivariantPocketEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # 2. Projekce globálního sekvenčního proteinového embeddingu (CLS token)
        self.protein_proj = nn.Sequential(
            nn.Linear(full_protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Stack Self-Attention Transformer vrstev
        self.num_attn_layers = num_attn_layers
        self.attn_layers = nn.ModuleList([
            TransformerSelfAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_attn_layers)
        ])
        
        # 4. Klasifikační hlava
        in_clf_dim = hidden_dim * 2 if use_pocket_pooling_in_head else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_clf_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode_pockets(self, mega_batch):
        """Pomocná metoda pro extrakci embeddingů kapes z PyG Mega-Batche."""
        return self.pocket_encoder(
            x=mega_batch.x,
            pos=mega_batch.pos,
            edge_index=mega_batch.edge_index,
            batch_idx=mega_batch.batch
        )

    def forward(self, mega_batch, protein_idx, full_protein_feats=None):
        """
        mega_batch: PyG Batch obsahující všechny 3D grafy kapes v minidávce
        protein_idx: [M] mapování každého kapsičkového grafu na index proteinu (0 .. B-1)
        full_protein_feats: [B, 1280] globální sekvenční embedding každého proteinu
        """
        device = mega_batch.x.device
        B = int(protein_idx.max().item() + 1) if protein_idx.numel() > 0 else 1
        
        # 1. Kódování všech 3D grafů kapes pomocí EGNN -> [M, hidden_dim]
        pocket_embs = self.encode_pockets(mega_batch)
        
        # 2. Uspořádání kapes do batchovaných sekvencí podle proteinů -> [B, K_max, hidden_dim]
        counts = torch.bincount(protein_idx, minlength=B)
        K_max = int(counts.max().item()) if counts.numel() > 0 else 1
        
        padded_pockets = torch.zeros(B, K_max, self.hidden_dim, device=device)
        pocket_mask = torch.ones(B, K_max, dtype=torch.bool, device=device) # True = padding
        
        for b in range(B):
            mask_b = (protein_idx == b)
            n_p = mask_b.sum().item()
            if n_p > 0:
                cur_pockets = pocket_embs[mask_b]
                
                # Stochastic Pocket Drop pro regularizaci během tréninku
                if self.training and n_p > 1 and self.pocket_drop_prob > 0:
                    keep_mask = torch.rand(n_p, device=device) > self.pocket_drop_prob
                    if keep_mask.sum() == 0:
                        keep_mask[torch.randint(0, n_p, (1,))] = True
                    cur_pockets = cur_pockets[keep_mask]
                    n_p = cur_pockets.size(0)
                    
                padded_pockets[b, :n_p] = cur_pockets
                pocket_mask[b, :n_p] = False
                
        # 3. Příprava sekvenčního CLS tokenu z plného proteinu
        if full_protein_feats is not None:
            cls_token = self.protein_proj(full_protein_feats).unsqueeze(1) # [B, 1, hidden_dim]
        else:
            # Fallback: průměr z dostupných kapes
            valid_counts = counts.clamp(min=1).view(B, 1, 1)
            cls_token = (padded_pockets * (~pocket_mask).unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts
            
        # 4. Spojení sekvence: [CLS_protein, Pocket_1, ..., Pocket_K]
        # Tvar: [B, K_max + 1, hidden_dim]
        x = torch.cat([cls_token, padded_pockets], dim=1)
        
        # Maska: CLS token (index 0) nikdy není padding (False)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, pocket_mask], dim=1) # [B, K_max + 1]
        
        # 5. Průchod Self-Attention vrstvami
        last_attn_weights = None
        for layer in self.attn_layers:
            x, last_attn_weights = layer(x, key_padding_mask=full_mask)
            
        # 6. Extrakce reprezentace pro klasifikaci
        cls_out = x[:, 0, :] # [B, hidden_dim]
        
        if self.use_pocket_pooling_in_head and K_max > 0:
            # Vypočteme průměr z aktualizovaných kapes (ignorujeme padding)
            pockets_out = x[:, 1:, :] # [B, K_max, hidden_dim]
            valid_pockets_mask = (~pocket_mask).unsqueeze(-1).float() # [B, K_max, 1]
            pockets_sum = (pockets_out * valid_pockets_mask).sum(dim=1)
            pockets_mean = pockets_sum / (~pocket_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
            
            combined_rep = torch.cat([cls_out, pockets_mean], dim=-1) # [B, hidden_dim * 2]
        else:
            combined_rep = cls_out
            
        logits = self.classifier(combined_rep) # [B, num_classes]
        
        return logits, last_attn_weights

# Alias pro zpětnou kompatibilitu a snadné importy
EGNNSelfAttentionMIL = EGNN_Self_Attention_MIL
