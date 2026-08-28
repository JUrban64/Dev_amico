import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Canonical SMILES pro 5 cílových kofaktorů v AMICO projektu
COFACTOR_SMILES = {
    'acetyl-CoA': r'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(=O)(O)O)n2cnc3c(N)ncnc23)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C',
    'ATP': r'Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'B12': r'CC1=CC2=C(C=C1C)[N+](=CN2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)(O)O[C@H](C)CNC(=O)CC[C@@]4([C@H]([C@@H]5[C@]6([C@@]([C@@H](C(=N6)/C(=C\7/[C@@]([C@@H](/C(=C/C8=N/C(=C(\C4=N5)/C)/[C@H](C8(C)C)CCC(=N)[O-])/N7)CCC(=N)[O-])(C)CC(=O)N)/C)CCC(=N)[O-])(C)CC(=O)N)C)CC(=O)N)C)O.[C]#N.[Co+2]',
    'FAD': r'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]4O[C@H](n5cnc6c(N)ncnc65)[C@H](O)[C@@H]4O)c2cc1C',
    'NAD': r'NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1'
}

COFACTOR_ORDER = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def get_cofactor_ecfps(n_bits=1024, radius=2):
    """
    Vygeneruje ECFP4 (Morgan) fingerprinty pro 5 kofaktorů.
    Pokud je k dispozici RDKit, spočítá je dynamicky. Jinak použije deterministický fallback.
    """
    fps = []
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        for name in COFACTOR_ORDER:
            smi = COFACTOR_SMILES[name]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Chyba při parsování SMILES pro {name}")
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(arr)
        return torch.tensor(np.stack(fps), dtype=torch.float32)
    except ImportError:
        # Deterministický fallback pro případ, že RDKit není v prostředí nainstalován
        print("Upozornění: RDKit není nainstalován, používám deterministický hash ECFP fallback.")
        for i, name in enumerate(COFACTOR_ORDER):
            np.random.seed(42 + i * 100)
            arr = (np.random.rand(n_bits) < 0.08).astype(np.float32)
            fps.append(arr)
        return torch.tensor(np.stack(fps), dtype=torch.float32)


class LigandCrossAttentionMIL(nn.Module):
    """
    Ligand-Protein Cross-Attention MIL model (Compatibility / Matching Architecture):
    
    1. Chemická reprezentace: Všech 5 kofaktorů je reprezentováno svými ECFP4 fingerprinty (1024-dim).
    2. Proteinové kapsy: Předpovězené vazebné kapsy (ESM-2, 1280-dim) a full-protein kontext.
    3. Cross-Attention: 5 chemických dotazů (Queries = Ligandy) prohledává kapsy proteinu (Keys & Values).
    4. Kompatibilita: Pro každý kofaktor se predikuje finální skóre afinity/kompatibility [B, 5].
    """
    def __init__(self, 
                 feature_dim=1280, 
                 ecfp_dim=1024, 
                 hidden_dim=256, 
                 num_heads=4, 
                 num_classes=5, 
                 dropout=0.2):
        super(LigandCrossAttentionMIL, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1. Projekce kapes a full-proteinu do společného latentního prostoru
        self.pocket_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Projekce ECFP chemických fingerprintů ligandů do stejného latentního prostoru
        self.ligand_proj = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Cross-Attention: Ligandy (Queries) se dotazují na Kapsy (Keys, Values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network po attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 4. Scoring Head: Mapuje interakční vektor každého ligandu na kompatibilitní skalár
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uložení fixních ECFP fingerprintů do bufferu modelu
        cofactor_fps = get_cofactor_ecfps(n_bits=ecfp_dim, radius=2)
        self.register_buffer('cofactor_fps', cofactor_fps) # [num_classes, ecfp_dim]
        
    def forward(self, pocket_features, padding_mask, full_protein_feature=None):
        """
        Args:
            pocket_features: [B, N, 1280] (embeddingy N kapes pro každý protein v batchi)
            padding_mask: [B, N] (True = padding kapsa k ignorování)
            full_protein_feature: [B, 1280] (globální embedding celého proteinu)
            
        Returns:
            logits: [B, 5] (skóre pro CrossEntropyLoss)
            attn_weights: [B, 5, N+1] (pozornost každého ze zkoumaných 5 ligandů ke kapsám)
        """
        B, N, _ = pocket_features.size()
        
        # 1. Projekce kapes na Keys a Values: [B, N, hidden_dim]
        k = self.pocket_proj(pocket_features)
        v = k
        
        # 2. Přidání full-proteinu jako globálního kontextového tokenu na pozici 0
        if full_protein_feature is not None:
            prot_tok = self.protein_proj(full_protein_feature).unsqueeze(1) # [B, 1, hidden_dim]
            k = torch.cat([prot_tok, k], dim=1) # [B, N+1, hidden_dim]
            v = torch.cat([prot_tok, v], dim=1) # [B, N+1, hidden_dim]
            
            # Aktualizace masky (pozice 0 je reálný protein, takže False)
            prot_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([prot_mask, padding_mask], dim=1) # [B, N+1]
        else:
            mask = padding_mask
            
        # 3. Projekce 5 kandidátních kofaktorů na Queries: [5, hidden_dim] -> [B, 5, hidden_dim]
        q_ligands = self.ligand_proj(self.cofactor_fps).unsqueeze(0).expand(B, -1, -1)
        
        # 4. Cross-Attention: Každý ligand hledá chemicky kompatibilní kapsu
        attn_out, attn_weights = self.cross_attn(
            query=q_ligands, 
            key=k, 
            value=v, 
            key_padding_mask=mask
        ) # attn_out: [B, 5, hidden_dim], attn_weights: [B, 5, N+1]
        
        # 5. Residual connection + FFN
        out = self.norm1(attn_out + q_ligands)
        out = self.norm2(out + self.ffn(out))
        
        # 6. Vyhodnocení skóre pro každou dvojici (Protein, Ligand): [B, 5, 1] -> [B, 5]
        logits = self.scorer(out).squeeze(-1)
        
        return logits, attn_weights
