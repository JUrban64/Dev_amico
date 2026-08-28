import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool

# Canonical SMILES pro 5 kofaktorů
COFACTOR_SMILES = {
    'acetyl-CoA': r'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(=O)(O)O)n2cnc3c(N)ncnc23)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C',
    'ATP': r'Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'B12': r'CC1=CC2=C(C=C1C)[N+](=CN2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)(O)O[C@H](C)CNC(=O)CC[C@@]4([C@H]([C@@H]5[C@]6([C@@]([C@@H](C(=N6)/C(=C\7/[C@@]([C@@H](/C(=C/C8=N/C(=C(\C4=N5)/C)/[C@H](C8(C)C)CCC(=N)[O-])/N7)CCC(=N)[O-])(C)CC(=O)N)/C)CCC(=N)[O-])(C)CC(=O)N)C)CC(=O)N)C)O.[C]#N.[Co+2]',
    'FAD': r'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]4O[C@H](n5cnc6c(N)ncnc65)[C@H](O)[C@@H]4O)c2cc1C',
    'NAD': r'NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1'
}
COFACTOR_ORDER = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def get_cofactor_ecfps(n_bits=1024, radius=2):
    """Vygeneruje ECFP4 fingerprinty pro 5 kofaktorů."""
    fps = []
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        for name in COFACTOR_ORDER:
            smi = COFACTOR_SMILES[name]
            mol = Chem.MolFromSmiles(smi)
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(arr)
        return torch.tensor(np.stack(fps), dtype=torch.float32)
    except ImportError:
        for i, name in enumerate(COFACTOR_ORDER):
            np.random.seed(42 + i * 100)
            arr = (np.random.rand(n_bits) < 0.08).astype(np.float32)
            fps.append(arr)
        return torch.tensor(np.stack(fps), dtype=torch.float32)

class GatedAttentionPool(nn.Module):
    """Gated Attention Pooling pro rezidua uvnitř kapsy."""
    def __init__(self, dim):
        super().__init__()
        self.attn_v = nn.Linear(dim, dim // 2)
        self.attn_u = nn.Linear(dim, dim // 2)
        self.attn_w = nn.Linear(dim // 2, 1)

    def forward(self, h, batch_idx):
        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        scores = self.attn_w(v * u).squeeze(-1) # [N]
        
        # Softmax per graph in batch
        # Pro zjednodušení v PyG použijeme scatter softmax
        from torch_geometric.utils import softmax
        weights = softmax(scores, batch_idx) # [N]
        out = global_mean_pool(h * weights.unsqueeze(-1) * batch_idx.bincount()[batch_idx].unsqueeze(-1), batch_idx)
        return out

class EquivariantPocketEncoder(nn.Module):
    """
    3D E(3)-Equivariant Graph Neural Network pro extrakci geometrických reprezentací kapes.
    Kombinuje ESM embeddingy reziduí s jejich 3D pozicemi a vzdálenostmi.
    """
    def __init__(self, node_dim=1280, hidden_dim=256, num_gnn_layers=2, dropout=0.35, coord_noise=0.05):
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

class EGNN_Ligand_Cross_Attention_MIL(nn.Module):
    """
    SOTA Architektura kombinující:
    1. 3D Equivariant Graph Neural Network (EGNN) pro geometrické kapsy
    2. Global Protein Context Injection
    3. Multi-Pocket Context Transformer
    4. Ligand Chemical Query Cross-Attention (ECFP4 Matching)
    5. Dual Compatibility & Temperature-Scaled Scoring Head
    """
    def __init__(self,
                 node_dim=1280,
                 full_protein_dim=1280,
                 ecfp_dim=1024,
                 hidden_dim=256,
                 num_gnn_layers=2,
                 num_heads=4,
                 num_classes=5,
                 dropout=0.35,
                 pocket_drop_prob=0.15):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pocket_drop_prob = pocket_drop_prob
        
        # 1. 3D EGNN Kodér kapes
        self.pocket_encoder = EquivariantPocketEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # 2. Projekce globálního proteinového embeddingu
        self.protein_proj = nn.Sequential(
            nn.Linear(full_protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Projekce chemických ligandových fingerprintů (ECFP4)
        self.ligand_proj = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. Multi-Pocket Self-Attention Transformer (kapsy komunikují mezi sebou a s celým proteinem)
        self.context_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        
        # 5. Ligand Cross-Attention: Ligandy (Queries) prohledávají Kapsy (Keys/Values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_norm1 = nn.LayerNorm(hidden_dim)
        self.cross_norm2 = nn.LayerNorm(hidden_dim)
        
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 6. Duální afinitní klasifikátor:
        # A) Nelineární MLP hlava
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # B) Learnable Temperature pro kosinové zarovnání
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        # Uložení fixních ECFP4 fingerprintů do bufferu
        cofactor_fps = get_cofactor_ecfps(n_bits=ecfp_dim, radius=2)
        self.register_buffer('cofactor_fps', cofactor_fps)

    def forward(self, mega_batch, protein_idx, full_protein_feats=None):
        """
        mega_batch: PyG Batch obsahující všechny 3D grafy kapes v batchi
        protein_idx: [M] mapování každého kapsičkového grafu na index proteinu v batchi (0 .. B-1)
        full_protein_feats: [B, 1280] volitelný globální embedding celého proteinu
        """
        device = mega_batch.x.device
        B = int(protein_idx.max().item() + 1) if protein_idx.numel() > 0 else 1
        
        # 1. Spuštění EGNN nad všemi 3D kapsami naráz (vektory [M, hidden_dim])
        pocket_embs = self.pocket_encoder(
            x=mega_batch.x,
            pos=mega_batch.pos,
            edge_index=mega_batch.edge_index,
            batch_idx=mega_batch.batch
        ) # [M, hidden_dim]
        
        # 2. Sestavení tensoru kapes pro jednotlivé proteiny: [B, K_max, hidden_dim]
        counts = torch.bincount(protein_idx, minlength=B)
        K_max = int(counts.max().item()) if counts.numel() > 0 else 1
        
        padded_pockets = torch.zeros(B, K_max, self.hidden_dim, device=device)
        pocket_mask = torch.ones(B, K_max, dtype=torch.bool, device=device) # True = padding
        
        # Vektorizované naplnění kapes s volitelným Stochastic Pocket Drop během tréninku
        for b in range(B):
            mask_b = (protein_idx == b)
            n_p = mask_b.sum().item()
            if n_p > 0:
                cur_pockets = pocket_embs[mask_b]
                
                # Stochastic Pocket Drop: pokud má protein více kapes, náhodně některé zamaskujeme
                if self.training and n_p > 1 and self.pocket_drop_prob > 0:
                    keep_mask = torch.rand(n_p, device=device) > self.pocket_drop_prob
                    if keep_mask.sum() == 0:  # Zajistíme, že alespoň jedna kapsa zůstane
                        keep_mask[torch.randint(0, n_p, (1,))] = True
                    cur_pockets = cur_pockets[keep_mask]
                    n_p = cur_pockets.size(0)
                    
                padded_pockets[b, :n_p] = cur_pockets
                pocket_mask[b, :n_p] = False
                
        # 3. Přidání globálního proteinového kontextu jako 1. token (CLS)
        if full_protein_feats is not None:
            prot_token = self.protein_proj(full_protein_feats).unsqueeze(1) # [B, 1, hidden_dim]
        else:
            # Fallback: průměr z přítomných kapes
            prot_token = (padded_pockets * (~pocket_mask).unsqueeze(-1)).sum(dim=1, keepdim=True) / counts.clamp(min=1).view(B, 1, 1)
            
        context_seq = torch.cat([prot_token, padded_pockets], dim=1) # [B, K_max+1, hidden_dim]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, pocket_mask], dim=1) # [B, K_max+1]
        
        # 4. Multi-Pocket Context Transformer
        context_seq = self.context_transformer(context_seq, src_key_padding_mask=full_mask) # [B, K_max+1, hidden_dim]
        
        # 5. Ligand Chemical Queries
        # 5 chemických dotazů promítnutých do latentního prostoru
        ligand_queries = self.ligand_proj(self.cofactor_fps).unsqueeze(0).expand(B, -1, -1) # [B, 5, hidden_dim]
        
        # 6. Ligand-Pocket Cross-Attention (Q = Ligandy, K/V = Kapsy + Protein)
        attn_out, attn_weights = self.cross_attn(
            query=ligand_queries,
            key=context_seq,
            value=context_seq,
            key_padding_mask=full_mask
        ) # [B, 5, hidden_dim]
        
        interact = self.cross_norm1(ligand_queries + attn_out)
        interact = self.cross_norm2(interact + self.cross_ffn(interact)) # [B, 5, hidden_dim]
        
        # 7. Duální skórování afinity:
        # A) Parametrické skóre z interakčního vektoru
        mlp_logits = self.scorer(interact).squeeze(-1) # [B, 5]
        
        # B) Kosinová geometrická shoda mezi ligandem a nejlepší kapsou
        q_norm = F.normalize(ligand_queries, p=2, dim=-1)
        z_norm = F.normalize(attn_out, p=2, dim=-1)
        cosine_sim = (q_norm * z_norm).sum(dim=-1) # [B, 5]
        
        final_logits = mlp_logits + self.temperature * cosine_sim # [B, 5]
        
        return final_logits, attn_weights
