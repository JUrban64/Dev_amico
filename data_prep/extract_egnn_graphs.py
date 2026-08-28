import json
import torch
import numpy as np
import argparse
import os
from torch_geometric.data import Data
from esm2_feature_ex import ESMFeatureExtractor
from tqdm import tqdm
from scipy.spatial.distance import cdist

def contact_map_to_edge_index(contact_map, threshold=8.0, coords=None):
    """
    Převede kontaktní matici (nebo souřadnice) na PyG edge_index [2, E]
    """
    if coords is not None and len(coords) > 0:
        dists = cdist(coords, coords)
        adj = (dists < threshold) & (dists > 0)
        row, col = np.where(adj)
    elif contact_map is not None and len(contact_map) > 0:
        cm = np.array(contact_map)
        np.fill_diagonal(cm, 0)
        row, col = np.where(cm > 0)
    else:
        return torch.empty((2, 0), dtype=torch.long)
        
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    return edge_index

def process_dataset(json_path, out_pt_path):
    print(f"Načítám dataset z {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if isinstance(raw_data, dict):
        data = []
        for pid, bs_info in raw_data.items():
            item = dict(bs_info)
            item['protein_id'] = pid
            data.append(item)
    else:
        data = raw_data
        
    extractor = ESMFeatureExtractor()
    dataset = []
    
    from Binding_site_ex import COFACTOR_FUNCTIONAL_GROUPS
    supported = list(COFACTOR_FUNCTIONAL_GROUPS.keys())
    
    for i, bs in enumerate(tqdm(data, desc="Extrahování 3D grafů (EGNN)")):
        seq = bs.get('binding_site_sequence', '')
        coords = bs.get('protein_coords', [])
        
        if not seq or not coords or len(coords) == 0:
            continue
            
        pid = bs.get('protein_id', f"unknown_{i}")
        pocket_id = bs.get('pocket_id', pid)
        if 'pocket' not in pid and 'pdb_file' in bs:
            pid = bs['pdb_file'].replace('.pdb', '') + f"_pocket_{i}"
            
        lbl = int(bs.get('label', -1))
        if lbl == -1:
            act = bs.get('actual_ligand_name', '')
            if act in supported:
                lbl = supported.index(act)
                
        if lbl == -1:
            continue
            
        try:
            # 1. Extrahování ESM-2 vlastností uzlů [L, 1280]
            emb = extractor.extract_embeddings(seq)
            x = torch.FloatTensor(emb)
            
            # 2. 3D souřadnice [L, 3]
            pos = torch.FloatTensor(coords)
            
            # Ošetření shody délek (truncation v ESM pokud sekvence > 1024)
            min_len = min(x.shape[0], pos.shape[0])
            x = x[:min_len]
            pos = pos[:min_len]
            
            # 3. Kontaktní hrany [2, E]
            contact_map = bs.get('contact_map', None)
            edge_index = contact_map_to_edge_index(contact_map, threshold=8.0, coords=pos.numpy())
            
            # 4. PyG Data objekt
            graph = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                protein_id=pid,
                pocket_id=pocket_id,
                label=torch.tensor(lbl, dtype=torch.long)
            )
            
            dataset.append(graph)
        except Exception as e:
            print(f"Chyba při zpracování {pid}: {e}")
            
    print(f"Extrahováno {len(dataset)} 3D grafů pro EGNN.")
    torch.save(dataset, out_pt_path)
    print(f"Uloženo do {out_pt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='p2rank_pockets_dataset.json', help='Vstupní JSON soubor')
    parser.add_argument('--output', default='egnn_dataset.pt', help='Výstupní PT soubor s PyG grafy')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)
