import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Statistiky velikosti kapes (počet reziduí) per class")
    parser.add_argument('--data-path', type=str, default='data_prep/esm_dataset.pt', help='Relativní cesta k datasetu z kořene projektu')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(base_dir, args.data_path)

    print(f"--- Načítám data z {data_path} ---")
    raw_data = torch.load(data_path, weights_only=False)

    # Názvy tříd
    target_names = {0: 'acetyl-CoA', 1: 'ATP', 2: 'B12', 3: 'FAD', 4: 'NAD'}

    records = []
    for item in raw_data:
        # Tenzor má rozměry [num_residues, 1280]
        num_residues = item['features'].shape[0]
        label_num = item['label']
        label_name = target_names.get(label_num, str(label_num))
        
        records.append({
            'Size': num_residues,
            'Class': label_name
        })

    df = pd.DataFrame(records)
    
    print(f"Celkem zpracováno kapes: {len(df)}")
    print("\nZákladní statistiky (počet reziduí) per class:")
    print(df.groupby('Class')['Size'].describe().to_string())

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(df['Class'].unique()))

    # --- Violin plot ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Class', y='Size', data=df, palette=palette, inner='quartile')
    plt.title("Distribuce velikosti kapes podle tříd (Violin Plot)")
    plt.ylabel("Velikost kapsy (počet reziduí)")
    plt.xlabel("Třída (Kofaktor)")
    violin_path = os.path.join(out_dir, "pocket_size_violin.png")
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Box plot ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Size', data=df, palette=palette)
    plt.title("Distribuce velikosti kapes podle tříd (Box Plot)")
    plt.ylabel("Velikost kapsy (počet reziduí)")
    plt.xlabel("Třída (Kofaktor)")
    box_path = os.path.join(out_dir, "pocket_size_boxplot.png")
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGrafy uloženy do:\n- {violin_path}\n- {box_path}")

if __name__ == '__main__':
    main()
