import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from dataset import load_split_ids
from tqdm import tqdm

def load_egnn_cross_bags(data_path, full_proteins_path):
    """
    Načte PyG 3D grafy kapes a spojí je s full-protein embeddingy.
    """
    print(f"Načítám PyG 3D grafy z {data_path}...")
    raw_graphs = torch.load(data_path, weights_only=False)
    
    full_proteins = {}
    if full_proteins_path and os.path.exists(full_proteins_path):
        print(f"Načítám full protein embeddingy z {full_proteins_path}...")
        full_proteins = torch.load(full_proteins_path, weights_only=False)
    else:
        print(f"Upozornění: Soubor {full_proteins_path} nenalezen, poběží v režimu bez explicitního full proteinu.")
        
    bags_dict = defaultdict(list)
    labels_dict = {}
    
    for graph in tqdm(raw_graphs, desc="Seskupování grafů do proteinových bagů"):
        raw_pid = graph.protein_id
        base_name = os.path.basename(str(raw_pid))
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        bags_dict[pid].append(graph)
        labels_dict[pid] = graph.label.item() if isinstance(graph.label, torch.Tensor) else graph.label
        
    bag_list = []
    missing_full_count = 0
    
    for pid, graphs in bags_dict.items():
        # Fallback na nuly, pokud chybí full protein embedding
        if pid in full_proteins:
            full_feat = full_proteins[pid]
            if not isinstance(full_feat, torch.Tensor):
                full_feat = torch.tensor(full_feat, dtype=torch.float32)
        else:
            full_feat = torch.zeros(1280, dtype=torch.float32)
            missing_full_count += 1
            
        pyg_batch = Batch.from_data_list(graphs)
        label_tensor = torch.tensor(labels_dict[pid], dtype=torch.long)
        
        bag_list.append({
            'protein_id': pid,
            'batch_graph': pyg_batch,
            'full_protein_feature': full_feat,
            'label': label_tensor
        })
        
    print(f"\nCelkem proteinových bagů: {len(bag_list)}")
    if missing_full_count > 0:
        print(f"Informace: U {missing_full_count} proteinů chyběl explicitní full-protein embedding (nahrazen nulovým vektorem).")
        
    return bag_list

def match_id(pid, id_set):
    if pid in id_set:
        return True
    clean_pid = pid.replace('_MERGED', '').replace('.pdb', '')
    if clean_pid in id_set:
        return True
    for x in id_set:
        if x.replace('_MERGED', '').replace('.pdb', '') == clean_pid:
            return True
    return False

def get_egnn_cross_splits(data_path, full_proteins_path, base_dir, split_suffix='_mil_0.5', use_nr=False):
    """
    Načte spojený dataset a rozdělí jej na train, val a test podle split souborů.
    """
    bags = load_egnn_cross_bags(data_path, full_proteins_path)
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=split_suffix, use_nr=use_nr)
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        print(f"Používám rozdělení ze split souborů (suffix: '{split_suffix}')...")
        for b in bags:
            pid = b['protein_id']
            if match_id(pid, train_ids):
                train_bags.append(b)
            elif match_id(pid, val_ids):
                val_bags.append(b)
            elif match_id(pid, test_ids):
                test_bags.append(b)
            else:
                pass
    else:
        print("Split soubory nenalezeny. Používám náhodné rozdělení 80/10/10...")
        import numpy as np
        np.random.seed(42)
        np.random.shuffle(bags)
        n = len(bags)
        train_bags = bags[:int(n*0.8)]
        val_bags = bags[int(n*0.8):int(n*0.9)]
        test_bags = bags[int(n*0.9):]
        
    print(f"Rozdělení -> Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    return train_bags, val_bags, test_bags

def egnn_cross_collate_fn(batch):
    """
    Vektorizovaná collate funkce pro EGNN + Ligand Cross Attention.
    """
    labels = torch.cat([item['label'].view(1) for item in batch])
    full_prot_feats = torch.stack([item['full_protein_feature'] for item in batch])
    
    all_graphs = []
    protein_indices = []
    
    for i, item in enumerate(batch):
        graphs = item['batch_graph'].to_data_list()
        all_graphs.extend(graphs)
        protein_indices.extend([i] * len(graphs))
        
    mega_batch = Batch.from_data_list(all_graphs)
    protein_idx = torch.tensor(protein_indices, dtype=torch.long)
    
    return mega_batch, protein_idx, full_prot_feats, labels
