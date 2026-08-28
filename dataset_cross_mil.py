import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from collections import defaultdict

def load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets'):
    """
    Načte ESM pocket embeddingy a ESM full protein embeddingy a spojí je.
    Očekává, že 'esm_full_proteins.pt' je dict {protein_id: tensor[1280]}.
    """
    print(f"Loading pocket features from {pockets_path}...")
    raw_pockets = torch.load(pockets_path, weights_only=False)
    
    print(f"Loading full protein features from {full_proteins_path}...")
    full_proteins = torch.load(full_proteins_path, weights_only=False)
    
    bags_dict = defaultdict(list)
    labels_dict = {}
    
    missing_full_prot = 0
    
    for item in raw_pockets:
        raw_pid = item['protein_id']
        base_name = os.path.basename(raw_pid)
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        # Ošetření: pokud chybí full_protein (např. chyba stahování), přeskočíme kapsu
        if pid not in full_proteins:
            missing_full_prot += 1
            continue
            
        feat = item['features'] # [num_residues, 1280]
        label = item['label']
        
        labels_dict[pid] = label
        
        if mode == 'pockets':
            feat = feat.mean(dim=0)
            bags_dict[pid].append(feat.numpy())
        else:
            bags_dict[pid].append(feat.numpy())
            
    if missing_full_prot > 0:
        print(f"Varování: U {missing_full_prot} kapes chyběl full_protein embedding. Tyto kapsy byly vyřazeny.")
        
    bag_list = []
    for pid in bags_dict:
        if mode == 'pockets':
            pocket_features = torch.FloatTensor(np.stack(bags_dict[pid]))
        else:
            pocket_features = torch.FloatTensor(np.concatenate(bags_dict[pid], axis=0))
            
        full_protein_feat = full_proteins[pid] # [1280]
        
        bag_list.append({
            'protein_id': pid,
            'pocket_features': pocket_features,       # [num_pockets, 1280]
            'full_protein_feature': full_protein_feat, # [1280]
            'label': torch.LongTensor([labels_dict[pid]])
        })
        
    print(f"Total bags (proteins) valid for Cross-Attention: {len(bag_list)}")
    return bag_list

class CrossMilDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item['pocket_features'], item['full_protein_feature'], item['label']

def custom_collate_fn(batch):
    """
    Collate function pro proměnlivý počet kapes.
    Zabalí pocket_features do padded sequence a vrátí masku.
    """
    pocket_feats = [item[0] for item in batch]
    full_prot_feats = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Pad pocket sequences
    lengths = torch.tensor([pf.shape[0] for pf in pocket_feats])
    max_len = lengths.max().item()
    
    dim = pocket_feats[0].shape[1]
    padded_pocket_feats = torch.zeros(len(batch), max_len, dim)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool) # True = padding (ignored in attention)
    
    for i, pf in enumerate(pocket_feats):
        l = pf.shape[0]
        padded_pocket_feats[i, :l, :] = pf
        mask[i, :l] = False # False = neignorovat
        
    full_prot_feats = torch.stack(full_prot_feats)
    labels = torch.cat(labels)
    
    return padded_pocket_feats, mask, full_prot_feats, labels
