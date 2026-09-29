import torch
from torch_geometric.data import Data, Batch
import numpy as np

from model import AttentionMIL_ESM
from model_self_attention_mil import SelfAttentionMIL
from model_cross_attention_mil import CrossAttentionMIL
from model_ligand_cross_attention_mil import LigandCrossAttentionMIL
from model_egnn_mil import EGNN_MIL_Classifier
from model_encoder_mil import EGNN_Encoder_MIL_Classifier
from model_egnn_ligand_cross_attention_mil import EGNN_Ligand_Cross_Attention_MIL
from model_egnn_self_attention_mil import EGNN_Self_Attention_MIL

print("--- TESTING ALL 8 NEURAL MODELS ---")

# 1. Dummy data for ESM models
B = 2
N = 4 # pockets per bag
D = 1280

pocket_feats = torch.randn(B, N, D)
padding_mask = torch.zeros(B, N, dtype=torch.bool)
padding_mask[0, 3] = True # one padded pocket
full_prot_feats = torch.randn(B, D)

# Standard MIL
m1 = AttentionMIL_ESM(in_features=1280, hidden_dim=256, num_classes=5, dropout=0.2, gated_attention=True)
out1, _ = m1(pocket_feats, padding_mask)
print(f"1. Standard MIL: Output shape = {out1.shape}")
assert out1.shape == (B, 5)

# Self-Attention MIL
m2 = SelfAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
out2, _ = m2(pocket_feats, padding_mask, full_prot_feats)
print(f"2. Self-Attention MIL: Output shape = {out2.shape}")
assert out2.shape == (B, 5)

# Cross-Attention MIL
m3 = CrossAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
out3, _ = m3(pocket_feats, padding_mask, full_prot_feats)
print(f"3. Cross-Attention MIL: Output shape = {out3.shape}")
assert out3.shape == (B, 5)

# Ligand Cross-Attention MIL
m4 = LigandCrossAttentionMIL(feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
out4, _ = m4(pocket_feats, padding_mask, full_prot_feats)
print(f"4. Ligand Cross-Attention MIL: Output shape = {out4.shape}")
assert out4.shape == (B, 5)

# 2. Dummy 3D graphs for EGNN models
# Bag 0 has 2 pockets, Bag 1 has 1 pocket
g1 = Data(x=torch.randn(15, 1280), pos=torch.randn(15, 3), edge_index=torch.randint(0, 15, (2, 30)), label=torch.tensor(0))
g2 = Data(x=torch.randn(20, 1280), pos=torch.randn(20, 3), edge_index=torch.randint(0, 20, (2, 40)), label=torch.tensor(0))
g3 = Data(x=torch.randn(12, 1280), pos=torch.randn(12, 3), edge_index=torch.randint(0, 12, (2, 24)), label=torch.tensor(1))

mega_batch = Batch.from_data_list([g1, g2, g3])
protein_idx = torch.tensor([0, 0, 1], dtype=torch.long)
full_prot_feats_egnn = torch.randn(2, 1280)

# EGNN Score-Level MIL
m5 = EGNN_MIL_Classifier(node_dim=1280, hidden_dim=128, num_gnn_layers=2, num_classes=5, dropout=0.3, gated_attention=True)
out5, _ = m5(mega_batch, protein_idx=protein_idx)
print(f"5. EGNN Score-Level MIL: Output shape = {out5.shape}")
assert out5.shape == (2, 5)

# EGNN Encoder MIL
m6 = EGNN_Encoder_MIL_Classifier(node_dim=1280, egnn_hidden_dim=128, mil_hidden_dim=128, num_gnn_layers=2, num_classes=5, dropout=0.3, num_heads=2, gated_attention=True)
out6, _ = m6(mega_batch, protein_idx=protein_idx)
print(f"6. EGNN Encoder MIL: Output shape = {out6.shape}")
assert out6.shape == (2, 5)

# EGNN Ligand Cross-Attention MIL
m7 = EGNN_Ligand_Cross_Attention_MIL(node_dim=1280, full_protein_dim=1280, ecfp_dim=1024, hidden_dim=128, num_gnn_layers=2, num_heads=4, num_classes=5, dropout=0.35, pocket_drop_prob=0.15)
out7, _ = m7(mega_batch, protein_idx, full_prot_feats_egnn)
print(f"7. EGNN Ligand Cross-Attention MIL: Output shape = {out7.shape}")
assert out7.shape == (2, 5)

# 8. EGNN Self-Attention MIL (EGNN pocket graphs + Sequence protein embedding + Self-Attention Transformer)
m8 = EGNN_Self_Attention_MIL(node_dim=1280, full_protein_dim=1280, hidden_dim=128, num_gnn_layers=2, num_heads=4, num_attn_layers=1, num_classes=5, dropout=0.3)
out8, attn8 = m8(mega_batch, protein_idx, full_prot_feats_egnn)
print(f"8. EGNN Self-Attention MIL: Output shape = {out8.shape}, Attention weights shape = {attn8.shape}")
assert out8.shape == (2, 5)
assert attn8.shape[0] == 2 # Batch size

print("\n>>> ALL 8 NEURAL MODELS PASSED SHAPE & FORWARD VERIFICATION! <<<")
