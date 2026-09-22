#!/usr/bin/env python3
"""
Analýza distribuce a velikosti klastrů napříč splity (Train, Validation, Test).

Funkcionalita:
1. Načte soubory klastrů (clusters{suffix}.json) a odpovídající splity (train/validation/test{suffix}.txt).
2. Pokud soubory neexistují, volitelně je dokáže vygenerovat spuštěním Foldseek shlukování (--run-clustering).
3. Vypočítá klíčové statistiky pro každý split:
   - Celkový počet proteinů a unikátních klastrů.
   - Průměrný počet proteinů v klastru (Mean ± Std).
   - Medián, IQR, Min a Max velikost klastru.
   - Zastoupení singletonů (|C| = 1), malých (2-5), středních (6-15) a velkých (>15) klastrů.
4. Spočítá detailní rozpad průměrné velikosti klastru podle jednotlivých tříd kofaktorů (ATP, NAD, FAD, B12, acetyl-CoA).
5. Provede Zero-Leakage validaci (ověření, že klastry nepřesahují mezi splity).
6. Analyzuje homogenitu klastrů (jedno-kofaktorové vs. multi-kofaktorové sdílené foldy).
7. Vypíše přehledné tabulky do konzole a vygeneruje publikační grafy (PNG) a exporty (CSV / JSON).
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Výchozí třídy v projektu AMICO
DEFAULT_TARGET_CLASSES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
CLASS_COLORS = {
    'acetyl-CoA': '#2E86AB',
    'ATP': '#A23B72',
    'B12': '#F18F01',
    'FAD': '#C73E1D',
    'NAD': '#3B1F2B'
}
SPLIT_COLORS = {
    'Train': '#2E86AB',
    'Validation': '#F18F01',
    'Test': '#6A994E',
    'Total': '#4A5568'
}

def find_default_structures_dir():
    """Pokusí se automaticky najít složku se strukturami."""
    candidates = [
        Path("./structures"),
        Path("./data_prep/structures"),
        Path("../structures"),
        Path("../PMCP/structures"),
        Path("/Users/jachymurban/Desktop/PMCP/structures"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            subdirs = [d.name for d in c.iterdir() if d.is_dir()]
            if any(cls in subdirs for cls in DEFAULT_TARGET_CLASSES):
                return c.resolve()
            if list(c.glob("*.pdb")) or list(c.glob("**/*.pdb")):
                return c.resolve()
    return None

def find_split_and_cluster_files(project_dir, suffix="_mil_0.5", data_prep_dir=None):
    """
    Vyhledá cluster JSON a split TXT soubory pro zadaný suffix.
    """
    if data_prep_dir is None:
        data_prep_dir = os.path.join(project_dir, "data_prep")
        
    search_dirs = [data_prep_dir, project_dir]
    
    clean_suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    
    cluster_file = None
    train_file = None
    val_file = None
    test_file = None
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        c_path = os.path.join(d, f"clusters{clean_suffix}.json")
        tr_path = os.path.join(d, f"train{clean_suffix}.txt")
        va_path = os.path.join(d, f"validation{clean_suffix}.txt")
        te_path = os.path.join(d, f"test{clean_suffix}.txt")
        
        if os.path.exists(c_path) and cluster_file is None:
            cluster_file = c_path
        if os.path.exists(tr_path) and train_file is None:
            train_file = tr_path
        if os.path.exists(va_path) and val_file is None:
            val_file = va_path
        if os.path.exists(te_path) and test_file is None:
            test_file = te_path
            
    return cluster_file, train_file, val_file, test_file

def load_clusters_data(cluster_file):
    """
    Načte soubor clusters.json.
    Struktura: { representative_id: [member_id, ...] }
    Vrací:
      - clusters_dict: { cluster_id: [members...] }
      - protein_to_cluster: { pid: cluster_id }
    """
    with open(cluster_file, 'r', encoding='utf-8') as f:
        raw_clusters = json.load(f)
        
    clusters_dict = {}
    protein_to_cluster = {}
    
    for c_idx, (rep, members) in enumerate(raw_clusters.items()):
        # Ujistíme se, že reprezentant je v seznamu členů
        member_set = set(members)
        member_set.add(rep)
        member_list = sorted(list(member_set))
        
        clusters_dict[c_idx] = {
            'representative': rep,
            'members': member_list,
            'size': len(member_list)
        }
        for m in member_list:
            protein_to_cluster[m] = c_idx
            
    return clusters_dict, protein_to_cluster

def load_splits_data(train_file, val_file, test_file):
    """
    Načte split TXT soubory.
    """
    splits = {}
    protein_to_split = {}
    
    for name, path in [('Train', train_file), ('Validation', val_file), ('Test', test_file)]:
        pids = []
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        pids.append(pid)
                        protein_to_split[pid] = name
        splits[name] = pids
        
    return splits, protein_to_split

def detect_protein_classes(proteins, structures_dir=None, project_dir=None):
    """
    Pokusí se dohledat kofaktorovou třídu pro každý protein:
    1. Z podsložek structures_dir (např. structures/ATP/*.pdb).
    2. Z názvu proteinu (pokud obsahuje jméno třídy).
    """
    class_map = {}
    
    # 1. Hledání v PDB souborech
    search_dirs = []
    if structures_dir and os.path.exists(structures_dir):
        search_dirs.append(Path(structures_dir))
    default_dir = find_default_structures_dir()
    if default_dir and default_dir not in search_dirs:
        search_dirs.append(default_dir)
        
    pdb_path_map = {}
    for d in search_dirs:
        for p in d.glob("**/*.pdb"):
            base_id = p.stem.replace('_MERGED', '')
            parts = p.parts
            detected_cls = None
            for part in reversed(parts):
                clean_p = part.replace('_prank_output', '')
                if clean_p in DEFAULT_TARGET_CLASSES:
                    detected_cls = clean_p
                    break
            if detected_cls:
                pdb_path_map[base_id] = detected_cls
                pdb_path_map[p.stem] = detected_cls
                
    for pid in proteins:
        clean_pid = pid.replace('_MERGED', '').replace('.pdb', '')
        if pid in pdb_path_map:
            class_map[pid] = pdb_path_map[pid]
        elif clean_pid in pdb_path_map:
            class_map[pid] = pdb_path_map[clean_pid]
        else:
            # Pokus z prefixu či suffixu
            found = None
            pid_lower = pid.lower().replace('-', '_')
            if 'acetyl' in pid_lower or 'coa' in pid_lower:
                found = 'acetyl-CoA'
            elif 'atp' in pid_lower:
                found = 'ATP'
            elif 'b12' in pid_lower or 'cobalamin' in pid_lower:
                found = 'B12'
            elif 'fad' in pid_lower:
                found = 'FAD'
            elif 'nad' in pid_lower:
                found = 'NAD'
            class_map[pid] = found if found else 'Unknown'
            
    return class_map

def compute_cluster_split_statistics(clusters_dict, protein_to_cluster, splits, class_map):
    """
    Vypočítá detailní statistiky klastrů pro jednotlivé splity a celkově.
    """
    split_stats = {}
    leakage_info = []
    
    # Ověření Zero Data Leakage
    cluster_to_splits = defaultdict(set)
    for split_name, pids in splits.items():
        for pid in pids:
            if pid in protein_to_cluster:
                c_idx = protein_to_cluster[pid]
                cluster_to_splits[c_idx].add(split_name)
                
    for c_idx, sp_set in cluster_to_splits.items():
        if len(sp_set) > 1:
            leakage_info.append({
                'cluster_id': c_idx,
                'representative': clusters_dict[c_idx]['representative'],
                'splits': list(sp_set),
                'size': clusters_dict[c_idx]['size']
            })

    # Výpočet pro každý split
    for split_name in ['Train', 'Validation', 'Test', 'Total']:
        if split_name == 'Total':
            pids_in_split = [p for p_list in splits.values() for p in p_list]
            if not pids_in_split:
                pids_in_split = list(protein_to_cluster.keys())
        else:
            pids_in_split = splits.get(split_name, [])
            
        c_indices = set()
        c_sizes = []
        cluster_members_in_split = defaultdict(list)
        
        for p in pids_in_split:
            if p in protein_to_cluster:
                c_idx = protein_to_cluster[p]
                c_indices.add(c_idx)
                cluster_members_in_split[c_idx].append(p)
                
        # Velikost klastrů v rámci daného splitu
        for c_idx, members in cluster_members_in_split.items():
            c_sizes.append(len(members))
            
        n_proteins = len(pids_in_split)
        n_clusters = len(c_indices)
        
        if c_sizes:
            sizes_arr = np.array(c_sizes)
            mean_size = float(np.mean(sizes_arr))
            std_size = float(np.std(sizes_arr))
            median_size = float(np.median(sizes_arr))
            q25, q75 = float(np.percentile(sizes_arr, 25)), float(np.percentile(sizes_arr, 75))
            min_size = int(np.min(sizes_arr))
            max_size = int(np.max(sizes_arr))
            
            singletons = int(np.sum(sizes_arr == 1))
            small = int(np.sum((sizes_arr >= 2) & (sizes_arr <= 5)))
            medium = int(np.sum((sizes_arr >= 6) & (sizes_arr <= 15)))
            large = int(np.sum(sizes_arr > 15))
        else:
            mean_size = std_size = median_size = q25 = q75 = 0.0
            min_size = max_size = singletons = small = medium = large = 0
            
        # Rozpad podle tříd
        class_breakdown = {}
        for cls in DEFAULT_TARGET_CLASSES:
            cls_pids = [p for p in pids_in_split if class_map.get(p) == cls]
            cls_clusters = set(protein_to_cluster[p] for p in cls_pids if p in protein_to_cluster)
            cls_c_sizes = [len([p for p in cls_pids if protein_to_cluster.get(p) == c]) for c in cls_clusters]
            
            cls_mean = float(np.mean(cls_c_sizes)) if cls_c_sizes else 0.0
            cls_std = float(np.std(cls_c_sizes)) if cls_c_sizes else 0.0
            cls_median = float(np.median(cls_c_sizes)) if cls_c_sizes else 0.0
            cls_max = int(np.max(cls_c_sizes)) if cls_c_sizes else 0
            
            class_breakdown[cls] = {
                'proteins': len(cls_pids),
                'clusters': len(cls_clusters),
                'mean_size': cls_mean,
                'std_size': cls_std,
                'median_size': cls_median,
                'max_size': cls_max
            }
            
        split_stats[split_name] = {
            'proteins': n_proteins,
            'clusters': n_clusters,
            'mean_size': mean_size,
            'std_size': std_size,
            'median_size': median_size,
            'q25': q25,
            'q75': q75,
            'min_size': min_size,
            'max_size': max_size,
            'singletons': singletons,
            'singletons_pct': (singletons / n_clusters * 100) if n_clusters > 0 else 0.0,
            'small': small,
            'small_pct': (small / n_clusters * 100) if n_clusters > 0 else 0.0,
            'medium': medium,
            'medium_pct': (medium / n_clusters * 100) if n_clusters > 0 else 0.0,
            'large': large,
            'large_pct': (large / n_clusters * 100) if n_clusters > 0 else 0.0,
            'raw_sizes': c_sizes,
            'class_breakdown': class_breakdown
        }
        
    # Homogenita klastrů (Single-class vs Multi-class)
    homo_count = 0
    hetero_count = 0
    for c_idx, data in clusters_dict.items():
        classes_in_c = set(class_map.get(m, 'Unknown') for m in data['members'])
        clean_classes = [c for c in classes_in_c if c != 'Unknown']
        if len(clean_classes) <= 1:
            homo_count += 1
        else:
            hetero_count += 1
            
    total_analyzed_clusters = len(clusters_dict)
    homogeneity = {
        'total_clusters': total_analyzed_clusters,
        'homogeneous': homo_count,
        'homogeneous_pct': (homo_count / total_analyzed_clusters * 100) if total_analyzed_clusters > 0 else 0.0,
        'heterogeneous': hetero_count,
        'heterogeneous_pct': (hetero_count / total_analyzed_clusters * 100) if total_analyzed_clusters > 0 else 0.0,
    }
    
    return split_stats, leakage_info, homogeneity

def print_ascii_tables(split_stats, leakage_info, homogeneity, suffix):
    """Vypíše přehledné formátované ASCII tabulky do konzole."""
    print("\n" + "=" * 88)
    print(f"📊 AMICO: STATISTIKA VELIKOSTI STRUKTURNÍCH KLASTRŮ (Split suffix: {suffix})")
    print("=" * 88)
    
    # 1. Hlavní tabulka velikostí klastrů napříč splity
    headers = [
        "Split", "Proteiny", "Klastry", "Průměr (Mean ± Std)", 
        "Medián (IQR)", "Min - Max", "Singleton (1)", "Malé (2-5)", "Velké (>15)"
    ]
    print(f"\n{headers[0]:<12s} | {headers[1]:<8s} | {headers[2]:<8s} | {headers[3]:<20s} | {headers[4]:<14s} | {headers[5]:<10s} | {headers[6]:<14s} | {headers[7]:<14s} | {headers[8]:<12s}")
    print("-" * 125)
    
    for split_name in ['Train', 'Validation', 'Test', 'Total']:
        st = split_stats[split_name]
        mean_str = f"{st['mean_size']:.2f} ± {st['std_size']:.2f}"
        med_str = f"{st['median_size']:.1f} ({st['q25']:.0f}-{st['q75']:.0f})"
        min_max_str = f"{st['min_size']} - {st['max_size']}"
        sing_str = f"{st['singletons']} ({st['singletons_pct']:.1f} %)"
        small_str = f"{st['small']} ({st['small_pct']:.1f} %)"
        large_str = f"{st['large']} ({st['large_pct']:.1f} %)"
        
        print(f"{split_name:<12s} | {st['proteins']:<8d} | {st['clusters']:<8d} | {mean_str:<20s} | {med_str:<14s} | {min_max_str:<10s} | {sing_str:<14s} | {small_str:<14s} | {large_str:<12s}")
    print("=" * 125)
    
    # 2. Rozpad průměrné velikosti klastru podle tříd
    print("\n🧬 PRŮMĚRNÝ POČET PROTEINŮ V KLASTRU DLE KLASTER/KOFAKTOR TŘÍDY:")
    print("-" * 105)
    print(f"{'Třída':<12s} | {'Train (Prot / Clust / Průměr)':<30s} | {'Val (Prot / Clust / Průměr)':<28s} | {'Test (Prot / Clust / Průměr)':<28s}")
    print("-" * 105)
    
    for cls in DEFAULT_TARGET_CLASSES:
        tr = split_stats['Train']['class_breakdown'].get(cls, {})
        va = split_stats['Validation']['class_breakdown'].get(cls, {})
        te = split_stats['Test']['class_breakdown'].get(cls, {})
        
        tr_s = f"{tr.get('proteins', 0):3d} p / {tr.get('clusters', 0):3d} c / {tr.get('mean_size', 0.0):.2f}"
        va_s = f"{va.get('proteins', 0):3d} p / {va.get('clusters', 0):3d} c / {va.get('mean_size', 0.0):.2f}"
        te_s = f"{te.get('proteins', 0):3d} p / {te.get('clusters', 0):3d} c / {te.get('mean_size', 0.0):.2f}"
        
        print(f"{cls:<12s} | {tr_s:<30s} | {va_s:<28s} | {te_s:<28s}")
    print("=" * 105)
    
    # 3. Diagnostika integrity (Leakage) a homogenity
    print("\n🛡️ VALIDACE INTEGRITY A KONTROLA DATA LEAKAGE:")
    if not leakage_info:
        print("  ✅ ZERO DATA LEAKAGE: Žádný klastr nezasahuje přes hranice splitů. Klastry jsou striktně disjunktní!")
    else:
        print(f"  ⚠️ POZOR: Detekován data leakage u {len(leakage_info)} klastrů!")
        for l in leakage_info[:5]:
            print(f"     - Klastr {l['cluster_id']} (rep: {l['representative']}, size: {l['size']}) ve splitech: {l['splits']}")
        if len(leakage_info) > 5:
            print(f"     - ... a dalších {len(leakage_info) - 5} klastrů.")
            
    print("\n🔬 HOMOGENITA KLASTRŮ (Strukturní klastry vs. kofaktorové třídy):")
    print(f"  🔹 Jednotřídní (Homogenní) klastry: {homogeneity['homogeneous']} ({homogeneity['homogeneous_pct']:.1f} %)")
    print(f"  🔹 Vícetřídní (Heterogenní) klastry: {homogeneity['heterogeneous']} ({homogeneity['heterogeneous_pct']:.1f} %)")
    if homogeneity['heterogeneous'] > 0:
        print("     (Sdílené strukturní motivy, např. Rossmann fold mezi různými nukleotidovými kofaktory)")
    print("-" * 88 + "\n")

def plot_cluster_size_distributions(split_stats, output_path):
    """
    Vygeneruje publikační graf velikostí klastrů (Boxplot + Kategorické zastoupení).
    """
    splits_to_plot = ['Train', 'Validation', 'Test', 'Total']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Panel: Boxplot velikostí klastrů
    box_data = [split_stats[s]['raw_sizes'] for s in splits_to_plot if split_stats[s]['raw_sizes']]
    valid_labels = [s for s in splits_to_plot if split_stats[s]['raw_sizes']]
    colors = [SPLIT_COLORS.get(s, '#2E86AB') for s in valid_labels]
    
    if box_data:
        bp = ax1.boxplot(
            box_data, 
            labels=valid_labels, 
            patch_artist=True, 
            showmeans=True,
            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, alpha=0.5, markeredgecolor='none', markerfacecolor='#4A5568')
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            
        ax1.set_title("Cluster Size Distribution Across Splits", fontsize=13, fontweight='bold', pad=12)
        ax1.set_xlabel("Split", fontsize=11, labelpad=8)
        ax1.set_ylabel("Proteins per Cluster (Log Scale)", fontsize=11)
        ax1.set_yscale('log')
        
        # Popisky průměru a mediánu nad boxy
        for i, s in enumerate(valid_labels):
            mean_val = split_stats[s]['mean_size']
            med_val = split_stats[s]['median_size']
            max_val = split_stats[s]['max_size']
            ax1.annotate(f"Mean: {mean_val:.1f}\nMed: {med_val:.0f}",
                         xy=(i + 1, max(max_val * 1.05, 2)),
                         xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='semibold')
            
        # Legenda
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Median'),
            Line2D([0], [0], marker='D', color='w', markeredgecolor='black', markerfacecolor='white', markersize=7, label='Mean')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9.5)
        
    # 2. Panel: Zastoupení velikostních kategorií (Stacked / Grouped Bar Chart)
    x = np.arange(len(valid_labels))
    width = 0.2
    
    tier_sing = [split_stats[s]['singletons_pct'] for s in valid_labels]
    tier_small = [split_stats[s]['small_pct'] for s in valid_labels]
    tier_med = [split_stats[s]['medium_pct'] for s in valid_labels]
    tier_large = [split_stats[s]['large_pct'] for s in valid_labels]
    
    r1 = ax2.bar(x - 1.5 * width, tier_sing, width, label='Singleton (1)', color='#2E86AB', edgecolor='black', alpha=0.9)
    r2 = ax2.bar(x - 0.5 * width, tier_small, width, label='Small (2–5)', color='#6A994E', edgecolor='black', alpha=0.9)
    r3 = ax2.bar(x + 0.5 * width, tier_med, width, label='Medium (6–15)', color='#F18F01', edgecolor='black', alpha=0.9)
    r4 = ax2.bar(x + 1.5 * width, tier_large, width, label='Large (>15)', color='#C73E1D', edgecolor='black', alpha=0.9)
    
    ax2.set_title("Cluster Size Tiers (% Share)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Split", fontsize=11, labelpad=8)
    ax2.set_ylabel("Share of Clusters (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_labels)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
    
    # Anotace na sloupcích
    for rects in [r1, r2, r3, r4]:
        for r in rects:
            h = r.get_height()
            if h > 3:
                ax2.annotate(f"{h:.0f}%",
                             xy=(r.get_x() + r.get_width() / 2, h),
                             xytext=(0, 2), textcoords="offset points",
                             ha='center', va='bottom', fontsize=7.5, fontweight='semibold')
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Graf distribuce velikostí klastrů uložen do: {output_path}")

def plot_class_cluster_sizes(split_stats, output_path):
    """
    Vygeneruje graf srovnání průměrné velikosti klastrů podle kofaktorů.
    """
    classes = DEFAULT_TARGET_CLASSES
    splits_to_compare = ['Train', 'Validation', 'Test']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    colors = [SPLIT_COLORS['Train'], SPLIT_COLORS['Validation'], SPLIT_COLORS['Test']]
    
    for i, s in enumerate(splits_to_compare):
        means = [split_stats[s]['class_breakdown'].get(c, {}).get('mean_size', 0.0) for c in classes]
        stds = [split_stats[s]['class_breakdown'].get(c, {}).get('std_size', 0.0) for c in classes]
        
        pos = x + (i - 1) * width
        bars = ax.bar(pos, means, width, yerr=stds, capsize=4, label=f"{s} Split", 
                      color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.1)
        
        for bar, m in zip(bars, means):
            if m > 0:
                ax.annotate(f"{m:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='semibold')
                
    ax.set_title("Mean Cluster Size by Cofactor Class Across Splits", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Cofactor Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Mean Proteins per Cluster", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight='semibold')
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Graf velikostí klastrů dle tříd uložen do: {output_path}")

def export_summary_table(split_stats, homogeneity, leakage_info, suffix, csv_path=None, json_path=None):
    """Exportuje statistiky do CSV a JSON."""
    rows = []
    for s in ['Train', 'Validation', 'Test', 'Total']:
        st = split_stats[s]
        row = {
            'Suffix': suffix,
            'Split': s,
            'Proteins': st['proteins'],
            'Clusters': st['clusters'],
            'Mean_Size': round(st['mean_size'], 2),
            'Std_Size': round(st['std_size'], 2),
            'Median_Size': round(st['median_size'], 2),
            'Q25': round(st['q25'], 2),
            'Q75': round(st['q75'], 2),
            'Min_Size': st['min_size'],
            'Max_Size': st['max_size'],
            'Singletons': st['singletons'],
            'Singletons_Pct': round(st['singletons_pct'], 2),
            'Small_2_5': st['small'],
            'Small_Pct': round(st['small_pct'], 2),
            'Medium_6_15': st['medium'],
            'Medium_Pct': round(st['medium_pct'], 2),
            'Large_gt15': st['large'],
            'Large_Pct': round(st['large_pct'], 2),
            'Leakage_Detected': len(leakage_info) > 0
        }
        for cls in DEFAULT_TARGET_CLASSES:
            c_info = st['class_breakdown'].get(cls, {})
            row[f'{cls}_Proteins'] = c_info.get('proteins', 0)
            row[f'{cls}_Clusters'] = c_info.get('clusters', 0)
            row[f'{cls}_Mean_Size'] = round(c_info.get('mean_size', 0.0), 2)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV tabulka uložena: {csv_path}")
        
    if json_path:
        full_json = {
            'suffix': suffix,
            'split_summary': rows,
            'homogeneity': homogeneity,
            'leakage_clusters_count': len(leakage_info),
            'leakage_clusters': leakage_info
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_json, f, indent=4)
        print(f"💾 JSON soubor uložen:  {json_path}")
        
    return df

def run_clustering_generation(tmscore=0.5, nr_threshold=None, target="structures", 
                              structures_dir=None, output_dir=None):
    """
    Vygeneruje clusters.json a train/val/test splity pomocí structure_clustering.py.
    """
    print(f"\n🚀 Spouštím shlukování Foldseek (TM-score práh: {tmscore}, NR: {nr_threshold})...")
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    
    try:
        from data_prep.structure_clustering import cluster_structures, find_foldseek_binary
        
        fs_bin = find_foldseek_binary()
        print(f"Nalezena binárka Foldseek: {fs_bin}")
        
        train, val, test, clusters = cluster_structures(
            target=target,
            tmscore_threshold=tmscore,
            nr_threshold=nr_threshold,
            structures_dir=structures_dir
        )
        
        if train is None or clusters is None:
            print("❌ Chyba: Shlukování se nezdařilo.")
            return None
            
        target_suffix = "_mil" if target == "structures" else ("_e3" if target == "binding_sites" else "_both")
        nr_sfx = f"_nr{nr_threshold}" if nr_threshold is not None else ""
        suffix = f"{target_suffix}_{tmscore}{nr_sfx}"
        
        save_dir = Path(output_dir) if output_dir else (project_root / "data_prep")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / f"train{suffix}.txt", "w") as f:
            f.write("\n".join(train) + "\n")
        with open(save_dir / f"validation{suffix}.txt", "w") as f:
            f.write("\n".join(val) + "\n")
        with open(save_dir / f"test{suffix}.txt", "w") as f:
            f.write("\n".join(test) + "\n")
        with open(save_dir / f"clusters{suffix}.json", "w") as f:
            json.dump(clusters, f, indent=4)
            
        print(f"✅ Shlukování dokončeno a soubory uloženy do {save_dir}:")
        print(f"   - train{suffix}.txt ({len(train)} proteinů)")
        print(f"   - validation{suffix}.txt ({len(val)} proteinů)")
        print(f"   - test{suffix}.txt ({len(test)} proteinů)")
        print(f"   - clusters{suffix}.json ({len(clusters)} klastrů)")
        
        return suffix
    except Exception as e:
        print(f"❌ Výjimka při spuštění shlukování: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Analýza velikosti a distribuce strukturních klastrů napříč splity v AMICO."
    )
    parser.add_argument(
        '--suffix', default='_mil_0.5',
        help="Přípona splitu a klastrů (např. _mil_0.5, _mil_0.3, _mil_0.7)."
    )
    parser.add_argument(
        '-c', '--clusters', default=None,
        help="Explicitní cesta k souboru clusters{suffix}.json."
    )
    parser.add_argument(
        '--train', default=None, help="Explicitní cesta k train{suffix}.txt."
    )
    parser.add_argument(
        '--val', default=None, help="Explicitní cesta k validation{suffix}.txt."
    )
    parser.add_argument(
        '--test', default=None, help="Explicitní cesta k test{suffix}.txt."
    )
    parser.add_argument(
        '-d', '--dir', '--structures-dir', dest='structures_dir', default=None,
        help="Cesta ke složce se strukturami pro přiřazení tříd proteinům."
    )
    parser.add_argument(
        '-o', '--output-dir', default='data_statistics',
        help="Cílová složka pro uložení výsledných statistik a grafů (default: data_statistics)."
    )
    parser.add_argument(
        '--run-clustering', action='store_true',
        help="Pokud klastry neexistují, automaticky spustí Foldseek structure_clustering.py pro jejich vygenerování."
    )
    parser.add_argument(
        '--tmscore', type=float, default=0.5,
        help="Práh TM-score pro případné spuštění nového shlukování (default: 0.5)."
    )
    parser.add_argument(
        '--nr-threshold', type=float, default=None,
        help="Práh Non-Redundant filtrace pro případné spuštění shlukování."
    )
    parser.add_argument(
        '--plot', default='cluster_size_distribution.png',
        help="Název souboru pro graf distribuce velikostí klastrů (default: cluster_size_distribution.png)."
    )
    parser.add_argument(
        '--class-plot', default='cluster_sizes_by_class.png',
        help="Název souboru pro graf velikostí dle tříd (default: cluster_sizes_by_class.png)."
    )
    parser.add_argument(
        '--csv', default='cluster_statistics.csv',
        help="Název CSV souboru pro export (default: cluster_statistics.csv)."
    )
    parser.add_argument(
        '--json', default='cluster_statistics.json',
        help="Název JSON souboru pro export (default: cluster_statistics.json)."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    clean_suffix = args.suffix if args.suffix.startswith("_") else f"_{args.suffix}"

    # 1. Hledání souborů
    cluster_file = args.clusters
    train_file = args.train
    val_file = args.val
    test_file = args.test
    
    if not cluster_file or not train_file or not val_file or not test_file:
        c_found, tr_found, va_found, te_found = find_split_and_cluster_files(
            str(project_root), suffix=clean_suffix
        )
        cluster_file = cluster_file or c_found
        train_file = train_file or tr_found
        val_file = val_file or va_found
        test_file = test_file or te_found

    # 2. Pokud soubory neexistují a je požadováno generování
    if (not cluster_file or not os.path.exists(cluster_file)) and args.run_clustering:
        print(f"⚠️ Soubor {cluster_file or f'clusters{clean_suffix}.json'} nenalezen. Spouštím shlukování...")
        gen_suffix = run_clustering_generation(
            tmscore=args.tmscore,
            nr_threshold=args.nr_threshold,
            structures_dir=args.structures_dir,
            output_dir=str(project_root / "data_prep")
        )
        if gen_suffix:
            clean_suffix = gen_suffix
            cluster_file, train_file, val_file, test_file = find_split_and_cluster_files(
                str(project_root), suffix=clean_suffix
            )

    if not cluster_file or not os.path.exists(cluster_file):
        print(f"❌ Chyba: Soubor s klastry pro suffix '{clean_suffix}' nebyl nalezen.")
        print("Můžete:")
        print("  1. Spustit shlukování s přepínačem --run-clustering:")
        print("     python data_statistics/analyze_cluster_distribution.py --run-clustering --tmscore 0.5")
        print("  2. Nebo nejprve vygenerovat klastry přes structure_clustering.py:")
        print("     python data_prep/structure_clustering.py --target structures --tmscore-threshold 0.5")
        sys.exit(1)

    print(f"🔍 Načítám klastry z: {cluster_file}")
    if train_file:
        print(f"🔍 Načítám Train split: {train_file}")
    if val_file:
        print(f"🔍 Načítám Val split:   {val_file}")
    if test_file:
        print(f"🔍 Načítám Test split:  {test_file}")

    # 3. Načtení dat
    clusters_dict, protein_to_cluster = load_clusters_data(cluster_file)
    splits, protein_to_split = load_splits_data(train_file, val_file, test_file)

    all_proteins = set(protein_to_cluster.keys()) | set(protein_to_split.keys())
    class_map = detect_protein_classes(
        all_proteins, structures_dir=args.structures_dir, project_dir=str(project_root)
    )

    # 4. Výpočet statistik
    split_stats, leakage_info, homogeneity = compute_cluster_split_statistics(
        clusters_dict, protein_to_cluster, splits, class_map
    )

    # 5. Výpis do terminálu
    print_ascii_tables(split_stats, leakage_info, homogeneity, clean_suffix)

    # 6. Uložení výsledků (CSV, JSON, Grafy)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (out_dir / args.csv) if args.csv else None
    json_path = (out_dir / args.json) if args.json else None
    export_summary_table(split_stats, homogeneity, leakage_info, clean_suffix, csv_path, json_path)

    if args.plot:
        plot_path = out_dir / args.plot
        try:
            plot_cluster_size_distributions(split_stats, plot_path)
        except Exception as e:
            print(f"⚠️ Nepodařilo se vygenerovat graf velikostí klastrů: {e}")

    if args.class_plot:
        class_plot_path = out_dir / args.class_plot
        try:
            plot_class_cluster_sizes(split_stats, class_plot_path)
        except Exception as e:
            print(f"⚠️ Nepodařilo se vygenerovat graf velikostí dle tříd: {e}")

if __name__ == '__main__':
    main()
