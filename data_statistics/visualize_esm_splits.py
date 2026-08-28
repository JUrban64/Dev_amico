import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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

def main():
    parser = argparse.ArgumentParser(description="2D UMAP / t-SNE projekce ESM embeddingů s rozlišením tříd a Train/Val/Test splitů")
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt', help='Cesta k datasetu kapes')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt', help='Cesta k full protein embeddingům')
    parser.add_argument('--split-suffix', default='mil_0.5', help='Suffix splitu (např. mil_0.5 nebo mil_0.7)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu')
    parser.add_argument('--method', choices=['umap', 'tsne'], default='umap' if UMAP_AVAILABLE else 'tsne')
    parser.add_argument('--sample-limit', type=int, default=5000, help='Limit počtu proteinů pro rychlost vykreslení (0 = všechny)')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)
    
    if not os.path.exists(pockets_path):
        print(f"Chyba: Soubor {pockets_path} neexistuje.")
        return
        
    use_full = os.path.exists(full_proteins_path)
    bags = load_cross_mil_data(pockets_path, full_proteins_path if use_full else pockets_path, mode='pockets')
    
    # Načtení splitů
    clean_suffix = args.split_suffix if args.split_suffix.startswith('mil_') or args.split_suffix.startswith('_') else f"mil_{args.split_suffix}"
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=clean_suffix, use_nr=args.use_nr)
    
    records = []
    for b in bags:
        pid = b['protein_id']
        lbl = b['label'].item()
        if use_full and 'full_protein_feature' in b:
            feat = b['full_protein_feature'].numpy()
        else:
            feat = b['pocket_features'].mean(dim=0).numpy()
            
        split_name = 'Unassigned'
        if match_id(pid, train_ids):
            split_name = 'Train'
        elif match_id(pid, val_ids):
            split_name = 'Validation'
        elif match_id(pid, test_ids):
            split_name = 'Test'
            
        records.append({
            'pid': pid,
            'feat': feat,
            'label': lbl,
            'Cofactor': TARGET_NAMES[lbl],
            'Split': split_name
        })
        
    if args.sample_limit > 0 and len(records) > args.sample_limit:
        print(f"Vybírám náhodný vzorek {args.sample_limit} proteinů pro přehlednost...")
        np.random.seed(42)
        idx_sample = np.random.choice(len(records), args.sample_limit, replace=False)
        records = [records[i] for i in idx_sample]
        
    X = np.array([r['feat'] for r in records])
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'feat'} for r in records])
    
    print(f"\n--- Redukce dimenzionality ({args.method.upper()}) pro {len(X)} vzorků ---")
    if args.method == 'umap':
        if not UMAP_AVAILABLE:
            print("UMAP není dostupný, přepínám na t-SNE.")
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
        
    coords_2d = reducer.fit_transform(X)
    df['Dim1'] = coords_2d[:, 0]
    df['Dim2'] = coords_2d[:, 1]
    
    # Vykreslení třípanelového grafu
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    
    palette = sns.color_palette("tab10", len(TARGET_NAMES))
    
    # 1. Panel: Všechny proteiny obarvené podle kofaktoru
    sns.scatterplot(
        data=df, 
        x='Dim1', y='Dim2', 
        hue='Cofactor', 
        palette=palette, 
        alpha=0.6, 
        s=30, 
        ax=axes[0]
    )
    axes[0].set_title("Strukturní prostor kofaktorů (ESM-2)", fontsize=13, fontweight='bold')
    axes[0].legend(title="Kofaktor", loc="best")
    
    # 2. Panel: Obarveno podle Train / Val / Test rozdělení
    split_palette = {'Train': '#4C72B0', 'Validation': '#55A868', 'Test': '#C44E52', 'Unassigned': '#CCCCCC'}
    sns.scatterplot(
        data=df[df['Split'] != 'Unassigned'], 
        x='Dim1', y='Dim2', 
        hue='Split', 
        palette=split_palette, 
        alpha=0.6, 
        s=30, 
        ax=axes[1]
    )
    axes[1].set_title(f"Rozmístění splitu '{clean_suffix}' v prostoru", fontsize=13, fontweight='bold')
    axes[1].legend(title="Sada", loc="best")
    
    # 3. Panel: Pouze Testovací proteiny (jak pokrývají prostor podle tříd)
    test_df = df[df['Split'] == 'Test']
    if len(test_df) > 0:
        sns.scatterplot(
            data=test_df, 
            x='Dim1', y='Dim2', 
            hue='Cofactor', 
            palette=palette, 
            alpha=0.85, 
            s=45, 
            ax=axes[2]
        )
        axes[2].set_title(f"Pouze Testovací sada ({len(test_df)} proteinů)", fontsize=13, fontweight='bold')
        axes[2].legend(title="Kofaktor", loc="best")
    else:
        axes[2].text(0.5, 0.5, "Žádné testovací proteiny", horizontalalignment='center')
        
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"esm_{args.method}_splits_{clean_suffix}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 2D mapový graf úspěšně uložen do: {out_file}")

if __name__ == '__main__':
    main()
