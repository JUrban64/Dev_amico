import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from egnn_pytorch import EGNN_Sparse
from model import AttentionMIL_ESM

class EGNNPocketEncoder(nn.Module):
    """
    EGNN Encoder pro zpracování jednoho kapsičkového 3D grafu.
    Aktualizuje jak vlastnosti uzlů, tak 3D souřadnice.
    Vrací pouze embeddingy kapes (ne logity).
    """
    def __init__(self, node_dim=1280, hidden_dim=128, num_gnn_layers=2, dropout=0.3):
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
        return pocket_emb

class EGNN_Encoder_MIL_Classifier(nn.Module):
    """
    Kombinovaný EGNN + Attention MIL model pro klasifikaci kofaktorů na úrovni proteinových bagů.
    Používá AttentionMIL_ESM z model.py pro zajištění stejné architektury a srovnatelnosti.
    """
    def __init__(self, node_dim=1280, egnn_hidden_dim=128, mil_hidden_dim=256, 
                 num_gnn_layers=2, num_classes=5, dropout=0.3, num_heads=1, 
                 attention_temp=1.0, gated_attention=True):
        super().__init__()
        
        # 1. EGNN Encoder pro 3D grafy kapes
        self.pocket_encoder = EGNNPocketEncoder(
            node_dim=node_dim,
            hidden_dim=egnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # 2. Stejný MIL modul jako v model.py (Embedding-level MIL)
        # Vstupem do MIL modulu je output z EGNN, proto in_features = egnn_hidden_dim
        self.mil = AttentionMIL_ESM(
            in_features=egnn_hidden_dim,
            hidden_dim=mil_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_heads=num_heads,
            attention_temp=attention_temp,
            gated_attention=gated_attention
        )

    def forward(self, batch_graphs, protein_idx=None):
        """
        Zpracuje PyG batch reprezentující kapsy VÍCE proteinů.
        batch_graphs: PyG Batch objekt všech kapes v batchi.
        protein_idx: Tenzor [M] určující, do kterého proteinu v batchi daná kapsa patří.
        """
        # 1. Získání embeddingu pro každou kapsu [M, egnn_hidden_dim]
        pocket_embs = self.pocket_encoder(
            x=batch_graphs.x,
            pos=batch_graphs.pos,
            edge_index=batch_graphs.edge_index,
            batch_idx=batch_graphs.batch
        )
        
        if protein_idx is None:
            protein_idx = torch.zeros(pocket_embs.size(0), dtype=torch.long, device=pocket_embs.device)
            
        from torch_geometric.utils import to_dense_batch
        
        # 2. Zarovnání (padding) kapes podle proteinů
        # dense_pocket_embs: [B, max_N, egnn_hidden_dim], mask: [B, max_N] (True pro validní kapsy)
        dense_pocket_embs, mask = to_dense_batch(pocket_embs, protein_idx)
        padding_mask = ~mask # True pro padding
        
        # 3. MIL modul z model.py (Embedding-level MIL agregace a klasifikace)
        logits, A = self.mil(dense_pocket_embs, padding_mask=padding_mask)
        
        return logits, A
