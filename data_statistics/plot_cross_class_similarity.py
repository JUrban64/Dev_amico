import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Přidání root adresáře do sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from dataset_cross_mil import load_cross_mil_data

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def compute_class_similarities(bags, use_full_protein=True):
    """
    Spočítá mezitřídní průměrné a maximální podobnosti mezi všemi 5 třídami kofaktorů.
    """
    class_vectors = {i: [] for i in range(len(TARGET_NAMES))}
    
    for b in bags:
        lbl = b['label'].item()
        if use_full_protein and 'full_protein_feature' in b:
            feat = b['full_protein_feature'].numpy()
        else:
            # Průměr přes kapsy
            feat = b['pocket_features'].mean(dim=0).numpy()
            
        class_vectors[lbl].append(feat)
        
    num_classes = len(TARGET_NAMES)
    mean_sim_matrix = np.zeros((num_classes, num_classes))
    top5_sim_matrix = np.zeros((num_classes, num_classes))
    
    print("\n--- Počítám mezitřídní podobnosti ---")
    for i in range(num_classes):
        vecs_i = np.array(class_vectors[i])
        print(f"Třída {TARGET_NAMES[i]:12s}: {len(vecs_i)} proteinů")
        
        for j in range(num_classes):
            vecs_j = np.array(class_vectors[j])
            if len(vecs_i) == 0 or len(vecs_j) == 0:
                continue
                
            sim_matrix = cosine_similarity(vecs_i, vecs_j)
            
            # Pro stejnou třídu ignorujeme diagonálu (identitu 1.0 se sebou samým)
            if i == j:
                np.fill_diagonal(sim_matrix, np.nan)
                mean_sim = np.nanmean(sim_matrix)
                # Top 5 % nejvyšších hodnot
                flat = sim_matrix[~np.isnan(sim_matrix)]
                top_cutoff = np.percentile(flat, 95) if len(flat) > 0 else 1.0
                top5_sim = np.mean(flat[flat >= top_cutoff]) if len(flat) > 0 else 1.0
            else:
                mean_sim = np.mean(sim_matrix)
                top_cutoff = np.percentile(sim_matrix.flatten(), 95)
                top5_sim = np.mean(sim_matrix[sim_matrix >= top_cutoff])
                
            mean_sim_matrix[i, j] = mean_sim
            top5_sim_matrix[i, j] = top5_sim
            
    return mean_sim_matrix, top5_sim_matrix

def plot_heatmaps(mean_sim_matrix, top5_sim_matrix, out_dir, prefix="esm"):
    os.makedirs(out_dir, exist_ok=True)
    
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Průměrná podobnost
    sns.heatmap(
        mean_sim_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="Blues", 
        xticklabels=TARGET_NAMES, 
        yticklabels=TARGET_NAMES,
        cbar_kws={'label': 'Průměrná kosinová podobnost'},
        ax=axes[0]
    )
    axes[0].set_title("Průměrná mezitřídní podobnost (ESM-2)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Cílový kofaktor", fontsize=11)
    axes[0].set_ylabel("Zdrojový kofaktor", fontsize=11)
    
    # 2. Top 5 % nejpodobnějších párů (zachytí sdílené nadrodiny / Rossmann fold)
    sns.heatmap(
        top5_sim_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="viridis", 
        xticklabels=TARGET_NAMES, 
        yticklabels=TARGET_NAMES,
        cbar_kws={'label': 'Top 5% maximální podobnost'},
        ax=axes[1]
    )
    axes[1].set_title("Maximální překryv rodin (Top 5% homologů)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Cílový kofaktor", fontsize=11)
    axes[1].set_ylabel("Zdrojový kofaktor", fontsize=11)
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"{prefix}_cross_class_similarity_heatmap.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Heatmapa úspěšně uložena do: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Vykreslení mezitřídní strukturní/sekvenční podobnosti (Heatmapa)")
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt', help='Cesta k datasetu kapes')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt', help='Cesta k full protein embeddingům')
    parser.add_argument('--feature-type', choices=['full_protein', 'pockets'], default='full_protein', help='Co porovnávat')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__)) # data_statistics
    
    pockets_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)
    
    if not os.path.exists(pockets_path):
        print(f"Chyba: Soubor {pockets_path} neexistuje.")
        return
        
    use_full = (args.feature_type == 'full_protein')
    if use_full and not os.path.exists(full_proteins_path):
        print(f"Varování: {full_proteins_path} nenalezen, přepínám na agregaci kapes.")
        use_full = False
        
    bags = load_cross_mil_data(pockets_path, full_proteins_path if use_full else pockets_path, mode='pockets')
    mean_sim, top5_sim = compute_class_similarities(bags, use_full_protein=use_full)
    plot_heatmaps(mean_sim, top5_sim, out_dir, prefix=f"esm_{args.feature_type}")

if __name__ == '__main__':
    main()
