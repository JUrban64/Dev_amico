import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Varování: Knihovna 'umap-learn' není nainstalována. UMAP projekce bude přeskočena.")

# Přidání kořenového adresáře projektu do sys.path pro import z dataset.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import load_data_from_tensors

def aggregate_features(bag_features, pooling='mean'):
    """
    Agreguje features kapes do jednoho vektoru pro protein.
    """
    if pooling == 'mean':
        return bag_features.mean(dim=0).numpy()
    elif pooling == 'max':
        return bag_features.max(dim=0).values.numpy()
    elif pooling == 'mean_max':
        mean_feat = bag_features.mean(dim=0).numpy()
        max_feat = bag_features.max(dim=0).values.numpy()
        return np.concatenate([mean_feat, max_feat])
    else:
        raise ValueError(f"Neznámá metoda pooling: {pooling}")

def main():
    parser = argparse.ArgumentParser(description="Projekce embeddingů do nižší dimenze (t-SNE, UMAP)")
    parser.add_argument('--data-path', type=str, default='data_prep/esm_dataset.pt', help='Relativní cesta k datasetu z kořene projektu')
    parser.add_argument('--mode', type=str, choices=['pockets', 'residues'], default='pockets', help='Mód načítání dat')
    parser.add_argument('--level', type=str, choices=['protein', 'pocket'], default='protein', help='Úroveň projekce (protein - 1 bod na protein, pocket - 1 bod na kapsu)')
    parser.add_argument('--pooling', type=str, choices=['mean', 'max', 'mean_max'], default='mean', help='Agregace kapes pokud je level=protein')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.dirname(os.path.abspath(__file__)) # data_statistics
    
    # Absolutní cesta k datům
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(base_dir, args.data_path)

    print("--- Načítání dat ---")
    bags = load_data_from_tensors(data_path, mode=args.mode)
    
    if len(bags) == 0:
        print("Chyba: Dataset je prázdný.")
        return

    X = []
    y = []

    print(f"\nPříprava dat na úrovni: {args.level}")
    for b in bags:
        if args.level == 'protein':
            feat = aggregate_features(b['features'], pooling=args.pooling)
            X.append(feat)
            y.append(b['label'].item())
        else:
            # level == 'pocket'
            features = b['features'].numpy()
            label = b['label'].item()
            for feat in features:
                X.append(feat)
                y.append(label)

    X = np.array(X)
    y_num = np.array(y)
    
    # Změna z číselných labelů na názvy tříd
    target_names = {0: 'acetyl-CoA', 1: 'ATP', 2: 'B12', 3: 'FAD', 4: 'NAD'}
    y = np.array([target_names.get(val, str(val)) for val in y_num])

    print(f"Celkový tvar dat pro projekci: {X.shape}")
    
    # Barvy podle tříd
    palette = sns.color_palette("husl", len(np.unique(y)))

    # --- t-SNE ---
    print("\n--- Počítám t-SNE ---")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette, alpha=0.7, legend='full')
    plt.title(f"t-SNE projekce embeddingů (level: {args.level})")
    tsne_path = os.path.join(out_dir, f'tsne_{args.level}.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE graf uložen do {tsne_path}")

    # --- UMAP ---
    if UMAP_AVAILABLE:
        print("\n--- Počítám UMAP ---")
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette=palette, alpha=0.7, legend='full')
        plt.title(f"UMAP projekce embeddingů (level: {args.level})")
        umap_path = os.path.join(out_dir, f'umap_{args.level}.png')
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"UMAP graf uložen do {umap_path}")

if __name__ == '__main__':
    main()
