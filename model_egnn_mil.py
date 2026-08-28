import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from egnn_pytorch import EGNN_Sparse

class EGNNPocketClassifier(nn.Module):
    """
    EGNN model pro zpracování jednoho kapsičkového 3D grafu.
    Funguje rovnou jako klasifikátor a vrací logity.
    """
    def __init__(self, node_dim=1280, hidden_dim=128, num_gnn_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
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
            )
            
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, pos, edge_index, batch_idx=None):
        """
        x: [N, 1280] vlastnosti uzlů
        pos: [N, 3] 3D souřadnice
        edge_index: [2, E] hrany
        batch_idx: [N] indexy grafů v batchi (pokud PyG Batch)
        """
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        feats = self.node_projection(x)
        
        # Centrování souřadnic vzhledem ke těžišti grafu (invariantní posun)
        mean_coors = global_mean_pool(pos, batch_idx)
        coors = pos - mean_coors[batch_idx]
        
        # Konkatenace pro egnn_pytorch
        x_in = torch.cat([coors, feats], dim=-1)
        
        for layer in self.gnn_layers:
            x_in = layer(
                x=x_in, 
                edge_index=edge_index, 
                batch=batch_idx
            )
            
        coors = x_in[:, :3]
        feats = x_in[:, 3:]
            
        feats = self.dropout(feats)
        # Globální pooling kapsy
        pocket_emb = global_mean_pool(feats, batch_idx)
        
        # Klasifikace rovnou na úrovni kapsy
        logits = self.classifier(pocket_emb)
        
        # Vracíme logity a embeddings (pro attention)
        return logits, pocket_emb


class EGNN_MIL_Classifier(nn.Module):
    """
    Kombinovaný EGNN + Attention MIL model pro klasifikaci kofaktorů na úrovni proteinových bagů.
    Využívá score-level MIL (agregace logitů z kapes pomocí attention).
    """
    def __init__(self, node_dim=1280, hidden_dim=128, num_gnn_layers=2, num_classes=5, dropout=0.3, gated_attention=True):
        super().__init__()
        
        self.gated_attention = gated_attention
        
        # 1. EGNN Klasifikátor pro 3D grafy kapes
        self.pocket_classifier = EGNNPocketClassifier(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 2. Gated Attention MIL Pooling přes kapsy jednoho proteinu
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        if self.gated_attention:
            self.attention_U = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Sigmoid()
            )
        self.attention_weights = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch_graphs, protein_idx=None):
        """
        Zpracuje PyG batch reprezentující kapsy VÍCE proteinů.
        batch_graphs: PyG Batch objekt všech kapes v batchi.
        protein_idx: Tenzor [K] určující, do kterého proteinu v batchi daná kapsa patří.
                     Pokud je None, předpokládá se, že všechny kapsy patří 1 proteinu.
        """
        # 1. Získání logitů a embeddingů pro každou kapsu
        pocket_logits, pocket_embs = self.pocket_classifier(
            x=batch_graphs.x,
            pos=batch_graphs.pos,
            edge_index=batch_graphs.edge_index,
            batch_idx=batch_graphs.batch
        )
        
        if protein_idx is None:
            protein_idx = torch.zeros(pocket_embs.size(0), dtype=torch.long, device=pocket_embs.device)
            
        from torch_geometric.utils import to_dense_batch
        
        # 2. Zarovnání (padding) kapes podle proteinů
        # dense_pocket_embs: [B, max_N, hidden_dim], mask: [B, max_N] (True pro validní kapsy)
        dense_pocket_embs, mask = to_dense_batch(pocket_embs, protein_idx)
        dense_pocket_logits, _ = to_dense_batch(pocket_logits, protein_idx) # [B, max_N, num_classes]
        
        padding_mask = ~mask # True pro padding
        
        # 3. Attention mechanismus přes kapsy
        A_V = self.attention_V(dense_pocket_embs)
        if self.gated_attention:
            A_U = self.attention_U(dense_pocket_embs)
            A_raw = self.attention_weights(A_V * A_U) # [B, max_N, 1]
        else:
            A_raw = self.attention_weights(A_V) # [B, max_N, 1]
            
        # Zmaskování paddingu
        A_raw = A_raw.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        
        A = torch.softmax(A_raw, dim=1) # Normalizace vah přes kapsy [B, max_N, 1]
        
        # 4. Vážený součet logitů kapes
        # A: [B, max_N, 1] -> transpose -> [B, 1, max_N]
        # bmm([B, 1, max_N], [B, max_N, num_classes]) -> [B, 1, num_classes]
        bag_logits = torch.bmm(A.transpose(1, 2), dense_pocket_logits).squeeze(1) # [B, num_classes]
        
        return bag_logits, A
