import torch
import os
from collections import defaultdict
from torch_geometric.data import Batch
from dataset import load_split_ids
from tqdm import tqdm

def load_egnn_bags(data_path):
    """
    Načte dataset PyG 3D grafů a seskupí je podle protein_id do PyG Batch objektů (bagů).
    """
    print(f"Načítám PyG 3D grafy z {data_path}...")
    raw_graphs = torch.load(data_path, weights_only=False)
    
    bags_dict = defaultdict(list)
    labels_dict = {}
    
    for graph in tqdm(raw_graphs, desc="Seskupování grafů do proteinových bagů"):
        raw_pid = graph.protein_id
        base_name = os.path.basename(str(raw_pid))
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        bags_dict[pid].append(graph)
        labels_dict[pid] = graph.label.item() if isinstance(graph.label, torch.Tensor) else graph.label
        
    bag_list = []
    for pid, graphs in bags_dict.items():
        # Vytvoření PyG Batch objektu pro všechny kapsy daného proteinu
        pyg_batch = Batch.from_data_list(graphs)
        label_tensor = torch.tensor(labels_dict[pid], dtype=torch.long)
        
        bag_list.append({
            'protein_id': pid,
            'batch_graph': pyg_batch,
            'label': label_tensor
        })
        
    print(f"\nCelkem proteinů (EGNN bags): {len(bag_list)}")
    total_pockets = sum([len(graphs) for graphs in bags_dict.values()])
    print(f"Celkem kapesních 3D grafů: {total_pockets}")
    if len(bag_list) > 0:
        print(f"Průměrně kapes na protein: {total_pockets / len(bag_list):.1f}")
        
    return bag_list

def get_egnn_splits(data_path, base_dir, split_suffix='_mil_0.5', use_nr=False):
    """
    Načte EGNN bagy a rozdělí je na train, validation a test podle dělení.
    """
    bags = load_egnn_bags(data_path)
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=split_suffix, use_nr=use_nr)
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        print(f"Používám rozdělení ze split souborů (suffix: {split_suffix})...")
        for b in bags:
            pid = b['protein_id']
            if pid in train_ids:
                train_bags.append(b)
            elif pid in val_ids:
                val_bags.append(b)
            elif pid in test_ids:
                test_bags.append(b)
            else:
                train_bags.append(b)
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
