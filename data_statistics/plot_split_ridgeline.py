import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from dataset import load_split_ids
from dataset_cross_mil import load_cross_mil_data

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

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

def analyze_split_similarities(bags, suffixes, base_dir, use_full_protein=True, use_nr=False, center_embeddings=True):
    """
    Pro každý split spočítá:
    1. Max podobnost k Train proteinům stejné třídy (Intra-class).
    2. Max podobnost k Train proteinům jiné třídy (Inter-class / Shared Fold).
    3. Max podobnost celkově.
    """
    all_feats = []
    protein_data = {}
    
    for b in bags:
        pid = b['protein_id']
        lbl = b['label'].item()
        if use_full_protein and 'full_protein_feature' in b:
            feat = b['full_protein_feature'].numpy()
        else:
            feat = b['pocket_features'].mean(dim=0).numpy()
            
        all_feats.append(feat)
        protein_data[pid] = {'feat': feat, 'label': lbl, 'class_name': TARGET_NAMES[lbl]}
        
    all_feats = np.array(all_feats)
    
    # Centrování pro odstranění Anisotropy (Embedding Cone Effectu)
    if center_embeddings and len(all_feats) > 0:
        mean_vector = np.mean(all_feats, axis=0, keepdims=True)
        print("Aplikuji centrování (odečtení střední hodnoty embeddingů) pro odstranění 0.88 baseline kuželu...")
        for pid in protein_data:
            centered_feat = protein_data[pid]['feat'] - mean_vector[0]
            # Renormalizace na jednotkovou délku pro korektní kosinovou podobnost
            norm = np.linalg.norm(centered_feat)
            protein_data[pid]['feat'] = centered_feat / (norm + 1e-8)
            
    records = []
    
    print("\n--- Počítám podobnosti napříč splity ---")
    for suffix in suffixes:
        clean_suffix = suffix if suffix.startswith('mil_') or suffix.startswith('_') else f"mil_{suffix}"
        train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=clean_suffix, use_nr=use_nr)
        
        if len(train_ids) == 0:
            print(f"Přeskakuji split '{suffix}': train soubor nenalezen.")
            continue
            
        train_pids = []
        train_vecs = []
        train_labels = []
        
        # Sestavení trénovací matice
        for pid, d in protein_data.items():
            if match_id(pid, train_ids):
                train_pids.append(pid)
                train_vecs.append(d['feat'])
                train_labels.append(d['label'])
                
        if len(train_vecs) == 0:
            continue
            
        train_mat = np.array(train_vecs)
        train_labels = np.array(train_labels)
        
        # Test proteiny
        test_pids = []
        test_vecs = []
        test_labels = []
        test_classes = []
        
        for pid, d in protein_data.items():
            if match_id(pid, test_ids):
                test_pids.append(pid)
                test_vecs.append(d['feat'])
                test_labels.append(d['label'])
                test_classes.append(d['class_name'])
                
        if len(test_vecs) == 0:
            continue
            
        test_mat = np.array(test_vecs)
        test_labels = np.array(test_labels)
        
        # Výpočet kosinové podobnosti mezi všemi Test a Train proteiny
        sim_matrix = cosine_similarity(test_mat, train_mat) # [num_test, num_train]
        
        for i in range(len(test_pids)):
            q_lbl = test_labels[i]
            
            same_mask = (train_labels == q_lbl)
            diff_mask = (train_labels != q_lbl)
            
            max_same = np.max(sim_matrix[i, same_mask]) if np.any(same_mask) else 0.0
            max_diff = np.max(sim_matrix[i, diff_mask]) if np.any(diff_mask) else 0.0
            max_all = np.max(sim_matrix[i])
            
            records.append({
                'Split': clean_suffix.replace('mil_', 'TM-práh '),
                'Protein_ID': test_pids[i],
                'Cofactor': test_classes[i],
                'Max_Sim_Same_Class': max_same,
                'Max_Sim_Diff_Class': max_diff,
                'Max_Sim_Overall': max_all
            })
            
    return pd.DataFrame(records)

def plot_ridgeline(df, out_dir, centered=True):
    os.makedirs(out_dir, exist_ok=True)
    if len(df) == 0:
        print("Chyba: Žádná data pro vykreslení.")
        return
        
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Violin graf rozdělení celkové podobnosti k Train sadě
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("Set2", len(df['Split'].unique()))
    
    sns.violinplot(
        data=df, 
        x='Split', 
        y='Max_Sim_Overall', 
        palette=palette,
        inner="quartile",
        cut=0
    )
    
    y_label = "Centrovaná kosinová podobnost k Train sadě" if centered else "Kosinová podobnost k Train sadě"
    plt.title(f"Distribuce ESM-2 podobnosti Test proteinů k Train sadě ({'Centrovaná' if centered else 'Raw'})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Práh klastrování splitu", fontsize=12, labelpad=10)
    plt.ylabel(y_label, fontsize=12, labelpad=10)
    
    out_file1 = os.path.join(out_dir, f"split_similarity_violin_{'centered' if centered else 'raw'}.png")
    plt.savefig(out_file1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Violin graf uložen do: {out_file1}")
    
    # 2. Porovnání Same-Class vs. Diff-Class (Odhalení zdroje podobnosti)
    df_melted = pd.melt(
        df,
        id_vars=['Split', 'Protein_ID', 'Cofactor'],
        value_vars=['Max_Sim_Same_Class', 'Max_Sim_Diff_Class'],
        var_name='Similarity_Type',
        value_name='Similarity'
    )
    df_melted['Similarity_Type'] = df_melted['Similarity_Type'].map({
        'Max_Sim_Same_Class': 'V rámci stejné třídy (Intra-class)',
        'Max_Sim_Diff_Class': 'K cizím třídám (Inter-class / Shared Fold)'
    })
    
    g = sns.catplot(
        data=df_melted,
        x='Cofactor',
        y='Similarity',
        hue='Similarity_Type',
        col='Split',
        kind='box',
        height=5,
        aspect=1.2,
        palette=['#4C72B0', '#C44E52'],
        fliersize=1.5
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"ESM-2 Podobnost: Stejná třída vs. Sdílené cizí rodiny ({'Centrovaná' if centered else 'Raw'})", fontsize=14, fontweight='bold')
    g.set_axis_labels("Kofaktor", y_label)
    
    out_file2 = os.path.join(out_dir, f"split_similarity_same_vs_diff_class_{'centered' if centered else 'raw'}.png")
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Boxplot srovnání Same vs. Diff class uložen do: {out_file2}")

def main():
    parser = argparse.ArgumentParser(description="Analýza a Ridgeline plot podobnosti mezi Train a Test sadami")
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt', help='Cesta k datasetu kapes')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt', help='Cesta k full protein embeddingům')
    parser.add_argument('--suffixes', nargs='+', default=['0.3', '0.5', '0.7', '0.9'], help='Seznam split suffixů k porovnání (např. 0.3 0.5 0.7 0.9)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant variantu')
    parser.add_argument('--raw', action='store_true', help='Vypnout centrování embeddingů')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)
    
    if not os.path.exists(pockets_path):
        print(f"Chyba: {pockets_path} nenalezen.")
        return
        
    use_full = os.path.exists(full_proteins_path)
    bags = load_cross_mil_data(pockets_path, full_proteins_path if use_full else pockets_path, mode='pockets')
    
    df = analyze_split_similarities(bags, args.suffixes, base_dir, use_full_protein=use_full, use_nr=args.use_nr, center_embeddings=not args.raw)
    plot_ridgeline(df, out_dir, centered=not args.raw)

if __name__ == '__main__':
    main()
