import torch
import numpy as np
import os
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

def load_data_from_tensors(data_path, mode='residues'):
    """
    Načte dataset přímo z pt souboru s čistými tenzory.
    Formát pt souboru by měl být list slovníků:
    [{'protein_id': str, 'features': Tensor(num_residues, 1280), 'label': int}, ...]
    
    mode: 'pockets' (zprůměruje rezidua do kapes) nebo 'residues' (nechá všechna rezidua).
    """
    print(f"Načítám čistá ESM data (mód: {mode}) z {data_path}...")
    
    raw_data = torch.load(data_path, weights_only=False)
    
    # Ukládáme si pro každý protein seznam embeddingů
    bags_dict = defaultdict(list)
    labels_dict = {}
    
    for item in tqdm(raw_data, desc="Zpracování bagů"):
        raw_pid = item['protein_id']
        # Extract just the basename in case raw_pid is a full path
        base_name = os.path.basename(raw_pid)
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        feat = item['features'] # Očekává se [N, 1280]
        label = item['label']
        
        labels_dict[pid] = label
        
        if mode == 'pockets':
            # Průměr přes rezidua v jedné kapse
            feat = feat.mean(dim=0)
            bags_dict[pid].append(feat.numpy())
        else:
            bags_dict[pid].append(feat.numpy())
            
    bag_list = []
    for pid in bags_dict:
        if mode == 'pockets':
            features = torch.FloatTensor(np.stack(bags_dict[pid]))
        else:
            features = torch.FloatTensor(np.concatenate(bags_dict[pid], axis=0))
            
        bag_list.append({
            'protein_id': pid,
            'features': features,
            'label': torch.LongTensor([labels_dict[pid]])
        })
        
    print(f"\nCelkem proteinů (bags): {len(bag_list)}")
    total_instances = sum([b['features'].shape[0] for b in bag_list])
    print(f"Celkem instancí (kapsy nebo rezidua celkem): {total_instances}")
    if len(bag_list) > 0:
        print(f"Průměrně instancí na bag: {total_instances / len(bag_list):.1f}")
        
    return bag_list

def load_split_ids(base_dir, split_suffix='_mil_0.5', use_nr=False):
    train_ids, val_ids, test_ids = set(), set(), set()
    if not split_suffix.startswith('_'):
        split_suffix = f'_{split_suffix}'

    if use_nr and '_nr' not in split_suffix:
        # Hledání odpovídajícího _nr split souboru
        nr_candidates = [
            f"{split_suffix}_nr0.95",
            f"{split_suffix}_nr0.9",
            f"{split_suffix}_nr"
        ]
        found = False
        for cand in nr_candidates:
            if os.path.exists(os.path.join(base_dir, f'data_prep/train{cand}.txt')):
                split_suffix = cand
                found = True
                break
        if not found:
            dp_dir = os.path.join(base_dir, 'data_prep')
            if os.path.exists(dp_dir):
                for f in os.listdir(dp_dir):
                    if f.startswith(f'train{split_suffix}_nr') and f.endswith('.txt'):
                        split_suffix = f.replace('train', '').replace('.txt', '')
                        found = True
                        break
        if not found:
            split_suffix = f"{split_suffix}_nr"

    train_path = os.path.join(base_dir, f'data_prep/train{split_suffix}.txt')
    val_path = os.path.join(base_dir, f'data_prep/validation{split_suffix}.txt')
    test_path = os.path.join(base_dir, f'data_prep/test{split_suffix}.txt')
    
    # Fallback to default _mil split if specific suffix file not found
    if not os.path.exists(train_path) and split_suffix != '_mil':
        train_path = os.path.join(base_dir, 'train_mil.txt')
        val_path = os.path.join(base_dir, 'validation_mil.txt')
        test_path = os.path.join(base_dir, 'test_mil.txt')
    
    if os.path.exists(train_path):
        train_ids = set(open(train_path).read().splitlines())
    if os.path.exists(val_path):
        val_ids = set(open(val_path).read().splitlines())
    if os.path.exists(test_path):
        test_ids = set(open(test_path).read().splitlines())
    return train_ids, val_ids, test_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='residues')
    args = parser.parse_args()
    bags = load_data_from_tensors(args.data_path, mode=args.mode)
