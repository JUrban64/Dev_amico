#!/usr/bin/env python3
"""
Skript pro výpočet statistik zastoupení jednotlivých tříd (kofaktorů) ze složky se strukturami (PDB).

Funkcionalita:
1. Prohledá složku 'structures' (automatická detekce nebo specifikace přes --dir).
2. Spočítá počet unikátních PDB struktur pro každou třídu (acetyl-CoA, ATP, B12, FAD, NAD i případné další).
3. Vypočítá procentuální podíly a míru nevyváženosti tříd (class imbalance ratio).
4. Analyzuje přítomnost P2Rank predikcí (*_prank_output, *_predictions.csv, *_pocket_*.pdb) a průměrný počet kapes na protein.
5. Vypíše přehlednou tabulku do konzole.
6. Volitelně vygeneruje publikační graf (PNG) a exportuje výsledky do CSV / JSON.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Podporované výchozí třídy v projektu AMICO
DEFAULT_TARGET_CLASSES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def find_default_structures_dir():
    """Pokusí se automaticky najít složku se strukturami."""
    candidates = [
        Path("./structures"),
        Path("../structures"),
        Path("../PMCP/structures"),
        Path("./data_prep/structures"),
        Path("../data_prep/structures"),
        Path("/Users/jachymurban/Desktop/PMCP/structures"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            # Ověříme, zda složka obsahuje nějaké pdb nebo podsložky tříd
            subdirs = [d.name for d in c.iterdir() if d.is_dir()]
            if any(cls in subdirs for cls in DEFAULT_TARGET_CLASSES):
                return c.resolve()
            if list(c.glob("*.pdb")) or list(c.glob("**/*.pdb")):
                return c.resolve()
    return None

def parse_p2rank_predictions_csv(csv_path):
    """Získá počet predikovaných kapes z _predictions.csv."""
    pocket_count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('name'):
                    continue
                pocket_count += 1
    except Exception:
        pass
    return pocket_count

def analyze_structures_directory(structures_dir, detailed=False):
    """
    Prohledá struktury a sestaví detailní statistiku po třídách.
    """
    structures_path = Path(structures_dir)
    if not structures_path.exists():
        raise FileNotFoundError(f"Složka neexistuje: {structures_path.resolve()}")

    print(f"📁 Prohledávám složku se strukturami: {structures_path.resolve()}")

    # Zjistíme, zda jsou struktury organizovány v podsložkách podle tříd
    subdirs = [d for d in structures_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    class_subdirs = {}
    for d in subdirs:
        clean_name = d.name.replace('_prank_output', '')
        class_subdirs[clean_name] = d

    # Data struktura pro statistiky
    class_stats = defaultdict(lambda: {
        'pdb_files': [],
        'prank_dirs': [],
        'pockets_per_protein': [],
        'has_prank_count': 0,
        'total_pockets': 0
    })

    # Režim A: Struktury jsou rozděleny v podsložkách podle tříd (např. structures/ATP/*.pdb)
    if any(c in class_subdirs for c in DEFAULT_TARGET_CLASSES) or len(class_subdirs) > 1:
        for class_name, dir_path in class_subdirs.items():
            # Hledání všech PDB souborů kromě vygenerovaných kapes P2Ranku
            all_pdbs = list(dir_path.glob("*.pdb"))
            # Filtrujeme případné _pocket_*.pdb
            full_pdbs = [p for p in all_pdbs if '_pocket_' not in p.name]
            class_stats[class_name]['pdb_files'].extend(full_pdbs)

            # Hledání P2Rank výstupů
            prank_dirs = list(dir_path.glob("*_prank_output"))
            class_stats[class_name]['prank_dirs'].extend(prank_dirs)

            # Mapování PDB -> P2Rank výstupy
            for pdb in full_pdbs:
                base_name = pdb.stem
                p_dir = dir_path / f"{base_name}_prank_output"
                if p_dir.exists() and p_dir.is_dir():
                    class_stats[class_name]['has_prank_count'] += 1
                    # Pokusíme se najít _predictions.csv
                    pred_csvs = list(p_dir.glob("*_predictions.csv"))
                    if pred_csvs:
                        n_pockets = parse_p2rank_predictions_csv(pred_csvs[0])
                    else:
                        # Fallback: spočítat _pocket_*.pdb
                        pocket_files = list(p_dir.glob("*_pocket_*.pdb"))
                        n_pockets = len(pocket_files)
                    class_stats[class_name]['pockets_per_protein'].append(n_pockets)
                    class_stats[class_name]['total_pockets'] += n_pockets
                else:
                    # Zkusit najít přímo kapsy ve stejné složce
                    sibling_pockets = list(dir_path.glob(f"{base_name}_pocket_*.pdb"))
                    if sibling_pockets:
                        class_stats[class_name]['has_prank_count'] += 1
                        class_stats[class_name]['pockets_per_protein'].append(len(sibling_pockets))
                        class_stats[class_name]['total_pockets'] += len(sibling_pockets)

    else:
        # Režim B: Plochá složka nebo rekurzivní vyhledávání
        print("  ℹ️ Detekována plochá nebo vnořená složka, prohledávám rekurzivně...")
        all_pdbs = list(structures_path.glob("**/*.pdb"))
        full_pdbs = [p for p in all_pdbs if '_pocket_' not in p.name and 'prank_output' not in str(p)]
        
        for p in full_pdbs:
            # Zkusit odvodit třídu z cesty (rodičovských složek)
            detected_class = "Unknown"
            for part in p.parts:
                clean_part = part.replace('_prank_output', '')
                if clean_part in DEFAULT_TARGET_CLASSES:
                    detected_class = clean_part
                    break
            class_stats[detected_class]['pdb_files'].append(p)

    return class_stats

def generate_summary_table(class_stats):
    """Sestaví souhrnný DataFrame se statistikami."""
    rows = []
    total_structures_all = sum(len(d['pdb_files']) for d in class_stats.values())
    total_pockets_all = sum(d['total_pockets'] for d in class_stats.values())

    # Seřadit podle výchozího pořadí tříd, ostatní na konec
    sorted_classes = sorted(
        class_stats.keys(),
        key=lambda c: DEFAULT_TARGET_CLASSES.index(c) if c in DEFAULT_TARGET_CLASSES else 999
    )

    for c in sorted_classes:
        data = class_stats[c]
        n_structs = len(data['pdb_files'])
        pct = (n_structs / total_structures_all * 100) if total_structures_all > 0 else 0.0
        n_prank = data['has_prank_count']
        prank_pct = (n_prank / n_structs * 100) if n_structs > 0 else 0.0
        tot_pockets = data['total_pockets']
        
        pockets_arr = data['pockets_per_protein']
        if pockets_arr:
            mean_p = np.mean(pockets_arr)
            std_p = np.std(pockets_arr)
            med_p = np.median(pockets_arr)
            min_p = np.min(pockets_arr)
            max_p = np.max(pockets_arr)
        else:
            mean_p, std_p, med_p, min_p, max_p = 0.0, 0.0, 0, 0, 0

        rows.append({
            'Class': c,
            'Structures': n_structs,
            'Share (%)': round(pct, 2),
            'P2Rank Done': n_prank,
            'P2Rank (%)': round(prank_pct, 1),
            'Total Pockets': tot_pockets,
            'Mean Pockets/Prot': round(mean_p, 2),
            'Std Pockets': round(std_p, 2),
            'Median Pockets': int(med_p),
            'Min Pockets': int(min_p),
            'Max Pockets': int(max_p)
        })

    df = pd.DataFrame(rows)
    return df, total_structures_all, total_pockets_all

def print_ascii_table(df, total_structures, total_pockets):
    """Vypíše přehlednou formátovanou tabulku do konzole."""
    print("\n" + "=" * 80)
    print(" 📊 STATISTIKY ZASTOUPENÍ TŘÍD V DATASETU STRUKTUR (AMICO)")
    print("=" * 80)

    # Zobrazení klíčových sloupců pro terminál
    display_df = df[['Class', 'Structures', 'Share (%)', 'P2Rank Done', 'Total Pockets', 'Mean Pockets/Prot']].copy()
    display_df.rename(columns={
        'Class': 'Třída (Kofaktor)',
        'Structures': 'Počet PDB',
        'Share (%)': 'Zastoupení',
        'P2Rank Done': 'P2Rank hotovo',
        'Total Pockets': 'Kapes celkem',
        'Mean Pockets/Prot': 'Průměr kapes'
    }, inplace=True)
    display_df['Zastoupení'] = display_df['Zastoupení'].apply(lambda x: f"{x:.2f} %")
    display_df['Průměr kapes'] = display_df['Průměr kapes'].apply(lambda x: f"{x:.2f}")

    print(display_df.to_string(index=False))
    print("-" * 80)

    # Celkový souhrn a Imbalance ratio
    print(f"🔹 Celkem PDB struktur:     {total_structures}")
    print(f"🔹 Celkem predikovaných kapes: {total_pockets}")
    
    struct_counts = df[df['Structures'] > 0]['Structures']
    if len(struct_counts) > 1:
        max_c = struct_counts.max()
        min_c = struct_counts.min()
        ratio = max_c / min_c if min_c > 0 else float('inf')
        print(f"🔹 Class Imbalance Ratio:   {ratio:.2f} : 1 (max={max_c} vs min={min_c})")
    print("=" * 80 + "\n")

def plot_class_distribution(df, output_path):
    """Vykreslí přehledný vizuální graf (Bar plot + Donut chart)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    classes = df['Class'].tolist()
    counts = df['Structures'].tolist()
    shares = df['Share (%)'].tolist()

    # Příjemná paleta barev
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A994E'][:len(classes)]

    # 1. Bar plot
    bars = ax1.bar(classes, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_title("Number of PDB Structures per Class", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Class (Cofactor)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Number of Structures", fontsize=11)

    # Popisky nad sloupci
    for bar, count, share in zip(bars, counts, shares):
        height = bar.get_height()
        ax1.annotate(f'{count}\n({share:.1f} %)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='semibold')

    # Nastavení horního limitu osy Y pro místo na popisky
    ax1.set_ylim(0, max(counts) * 1.18 if counts else 10)

    # 2. Donut chart
    wedges, texts, autotexts = ax2.pie(
        counts, 
        labels=classes, 
        autopct='%1.1f%%',
        startangle=140, 
        colors=colors,
        pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_color('black')
        at.set_fontweight('bold')
        at.set_fontsize(9.5)
    for t in texts:
        t.set_fontsize(10.5)

    ax2.set_title("Class Proportions in Dataset", fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Graf distribuce tříd uložen do: {output_path}")

def plot_pocket_statistics(df, output_path, class_stats=None):
    """Vykreslí přehledný vizuální graf statistik predikovaných kapes (P2Rank).
    Panel 1: Celkový počet kapes per třída s procentuálním podílem.
    Panel 2: Distribuce počtu kapes na protein (Boxplot s vyznačeným mediánem a průměrem).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    classes = df['Class'].tolist()
    total_pockets = df['Total Pockets'].tolist()

    total_all_pockets = sum(total_pockets)
    pocket_shares = [(p / total_all_pockets * 100) if total_all_pockets > 0 else 0.0 for p in total_pockets]

    # Příjemná paleta barev shodná s distribucí tříd
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A994E'][:len(classes)]

    # 1. Panel: Celkový počet kapes per třída
    bars1 = ax1.bar(classes, total_pockets, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_title("Total Predicted Pockets per Class", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Class (Cofactor)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Total Number of Pockets", fontsize=11)

    # Popisky nad sloupci
    for bar, count, share in zip(bars1, total_pockets, pocket_shares):
        height = bar.get_height()
        ax1.annotate(f'{count}\n({share:.1f} %)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='semibold')

    ax1.set_ylim(0, max(total_pockets) * 1.18 if total_pockets and max(total_pockets) > 0 else 10)

    # 2. Panel: Distribuce počtu kapes na protein (Boxplot)
    box_data = []
    if class_stats:
        for c in classes:
            pts = class_stats[c].get('pockets_per_protein', [])
            box_data.append(pts if pts else [0])
    else:
        # Fallback na hodnoty z df, pokud class_stats nejsou k dispozici
        box_data = [[df.loc[df['Class'] == c, 'Mean Pockets/Prot'].values[0]] for c in classes]

    bp = ax2.boxplot(
        box_data,
        positions=range(len(classes)),
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5.5),
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='#666666', markeredgecolor='none', markersize=2.5, alpha=0.3)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_title("Predicted Pockets per Protein", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Class (Cofactor)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Pockets per Protein", fontsize=11)

    # Legenda pro medián a průměr
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], marker='D', color='w', markeredgecolor='black', markerfacecolor='white', markersize=6, label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Graf statistik kapes uložen do: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analýza a statistiky zastoupení jednotlivých tříd ze složky se strukturami (AMICO)."
    )
    parser.add_argument(
        '-d', '--dir', '--structures-dir', dest='structures_dir', default=None,
        help="Cesta ke složce se strukturami (pokud není zadána, pokusí se najít automaticky)."
    )
    parser.add_argument(
        '-o', '--output-dir', default='data_statistics',
        help="Cílová složka pro uložení výsledných statistik a grafů (default: data_statistics)."
    )
    parser.add_argument(
        '--csv', default='class_distribution.csv',
        help="Název CSV souboru pro export tabulky (default: class_distribution.csv)."
    )
    parser.add_argument(
        '--json', default='class_distribution.json',
        help="Název JSON souboru pro export (default: class_distribution.json)."
    )
    parser.add_argument(
        '--plot', default='class_distribution.png',
        help="Název souboru grafu distribuce tříd (default: class_distribution.png, prázdný řetězec = nekreslit)."
    )
    parser.add_argument(
        '--pocket-plot', default='pocket_statistics.png',
        help="Název souboru grafu statistik kapes (default: pocket_statistics.png, prázdný řetězec = nekreslit)."
    )
    args = parser.parse_args()

    # 1. Hledání složky
    struct_dir = args.structures_dir
    if struct_dir is None:
        struct_dir = find_default_structures_dir()
        if struct_dir is None:
            print("❌ Chyba: Složka se strukturami nebyla automaticky nalezena.")
            print("Zadejte prosím cestu explicitně pomocí parametru:")
            print("  python data_statistics/count_class_distribution.py --dir /cesta/ke/structures")
            sys.exit(1)
        print(f"🔍 Automaticky detekována složka: {struct_dir}")

    # 2. Analýza
    class_stats = analyze_structures_directory(struct_dir)
    if not class_stats:
        print(f"⚠️ Ve složce {struct_dir} nebyly nalezeny žádné struktury.")
        sys.exit(0)

    # 3. Sestavení souhrnné tabulky
    df, total_structs, total_pockets = generate_summary_table(class_stats)

    # 4. Výpis do terminálu
    print_ascii_table(df, total_structs, total_pockets)

    # 5. Uložení výsledků
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = out_dir / args.csv
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV tabulka uložena: {csv_path}")

    if args.json:
        json_path = out_dir / args.json
        df.to_json(json_path, orient='records', indent=4)
        print(f"💾 JSON soubor uložen:  {json_path}")

    if args.plot:
        plot_path = out_dir / args.plot
        try:
            plot_class_distribution(df, plot_path)
        except Exception as e:
            print(f"⚠️ Nepodařilo se vygenerovat graf distribuce tříd: {e}")

    if args.pocket_plot:
        pocket_plot_path = out_dir / args.pocket_plot
        try:
            plot_pocket_statistics(df, pocket_plot_path, class_stats=class_stats)
        except Exception as e:
            print(f"⚠️ Nepodařilo se vygenerovat graf statistik kapes: {e}")

if __name__ == '__main__':
    main()
