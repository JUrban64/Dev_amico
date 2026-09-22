#!/usr/bin/env python3
"""
Master skript pro komplexní evaluaci vlivu Non-Redundant (NR) filtrace v projektu AMICO.

Postup:
1. Provede strukturní shlukování pro prahy TM-score: 0.3, 0.5, 0.7 a 0.9:
   - Varianta A (Standardní): Bez NR předfiltrace (_mil_<tmscore>).
   - Varianta B (Non-Redundant): S NR filtrací na prahu TM-score 0.95 (_mil_<tmscore>_nr0.95).
2. Spočítá a zobrazí rozdíly ve velikosti a počtu klastrů před a po NR filtraci.
3. Spustí benchmark všech modelů (benchmark_all_models.py) napříč všemi těmito splity.
4. Vyhodnotí a vykreslí rozdíly ve výkonu modelů (Δ Test Macro F1, Δ Accuracy) způsobené odstraněním homologů:
   - Výpis souhrnné srovnávací tabulky do konzole.
   - Publikační graf srovnání (PNG).
   - Export do CSV, JSON a Markdown reportu.
"""

import os
import sys
import glob
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_prep.structure_clustering import generate_clustering_splits, find_foldseek_binary
from data_statistics.analyze_cluster_distribution import (
    load_clusters_data, load_splits_data, detect_protein_classes,
    compute_cluster_split_statistics, DEFAULT_TARGET_CLASSES
)

# Výchozí sady prahů a modelů
DEFAULT_TMSCORES = [0.3, 0.5, 0.7, 0.9]
DEFAULT_NR_THRESHOLD = 0.95

ALL_MODEL_KEYS = [
    "foldseek", 
    "sequence_mlp",
    "residue_mil",
    "standard_mil", 
    "self_attention_mil", 
    "cross_attention_mil", 
    "encoder_mil",
    "ligand_cross_mil",
    "egnn_mil",
    "egnn_ligand_cross_mil"
]

def find_structures_directory(user_dir=None):
    """Najde složku se strukturami v projektu."""
    if user_dir and os.path.exists(user_dir):
        return Path(user_dir).resolve()
    candidates = [
        PROJECT_ROOT / "structures",
        PROJECT_ROOT / "data_prep" / "structures",
        PROJECT_ROOT.parent / "structures",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None

def stage_1_generate_splits(tmscores, nr_threshold=0.95, structures_dir=None, 
                            output_dir=None, force=False):
    """
    Fáze 1: Vygeneruje páry splitů (bez NR a s NR 0.95) pro všechny zadané prahy TM-score.
    """
    save_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data_prep")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("FÁZE 1: GENEROVÁNÍ STRUKTURNÍCH SPLITŮ (Foldseek Clustering)")
    print(f"Prahy TM-score: {tmscores} | NR práh: {nr_threshold}")
    print(f"Cílová složka pro splity: {save_dir}")
    print("=" * 80)

    generated_splits = []
    cluster_meta = {}

    for tmscore in tmscores:
        # 1. Varianta Bez NR
        sfx_no_nr = f"_mil_{tmscore}"
        c_file_no_nr = save_dir / f"clusters{sfx_no_nr}.json"
        tr_file_no_nr = save_dir / f"train{sfx_no_nr}.txt"

        if c_file_no_nr.exists() and tr_file_no_nr.exists() and not force:
            print(f"\n⚡ Split bez NR {sfx_no_nr} již existuje. Přeskakuji generování.")
        else:
            print(f"\n🔄 Generuji split bez NR (TM-score {tmscore})...")
            generate_clustering_splits(
                target="structures",
                tmscore_threshold=tmscore,
                nr_threshold=None,
                structures_dir=str(structures_dir) if structures_dir else None,
                output_dir=str(save_dir)
            )

        clean_name_no_nr = sfx_no_nr.lstrip("_")
        generated_splits.append(clean_name_no_nr)

        # 2. Varianta S NR
        sfx_nr = f"_mil_{tmscore}_nr{nr_threshold}"
        c_file_nr = save_dir / f"clusters{sfx_nr}.json"
        tr_file_nr = save_dir / f"train{sfx_nr}.txt"

        if c_file_nr.exists() and tr_file_nr.exists() and not force:
            print(f"\n⚡ Split s NR {sfx_nr} již existuje. Přeskakuji generování.")
        else:
            print(f"\n🔄 Generuji split s NR (TM-score {tmscore}, NR {nr_threshold})...")
            generate_clustering_splits(
                target="structures",
                tmscore_threshold=tmscore,
                nr_threshold=nr_threshold,
                structures_dir=str(structures_dir) if structures_dir else None,
                output_dir=str(save_dir)
            )

        clean_name_nr = sfx_nr.lstrip("_")
        generated_splits.append(clean_name_nr)

    print("\n✅ Všechny požadované splity jsou připraveny.")
    return generated_splits

def stage_2_compare_clusters(tmscores, nr_threshold=0.95, data_prep_dir=None, out_dir=None):
    """
    Fáze 2: Analýza rozdílů ve velikostech klastrů a redukci proteinů mezi Bez-NR a S-NR splity.
    """
    dp_dir = Path(data_prep_dir) if data_prep_dir else (PROJECT_ROOT / "data_prep")
    out_path = Path(out_dir) if out_dir else (PROJECT_ROOT / "data_statistics")
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print("FÁZE 2: SROVNÁNÍ VELIKOSTI KLASTRŮ A VLIVU NR FILTRACE (No-NR vs. NR 0.95)")
    print("=" * 95)

    comparison_rows = []

    print(f"\n{'TM-score':<10s} | {'Režim':<12s} | {'Proteiny (Tr/Va/Te/Celk)':<25s} | {'Klastry':<8s} | {'Průměr ± Std':<16s} | {'Medián':<8s} | {'Redukce (%)':<12s}")
    print("-" * 105)

    for tmscore in tmscores:
        sfx_no_nr = f"_mil_{tmscore}"
        sfx_nr = f"_mil_{tmscore}_nr{nr_threshold}"

        c_file_no_nr = dp_dir / f"clusters{sfx_no_nr}.json"
        tr_no_nr = dp_dir / f"train{sfx_no_nr}.txt"
        va_no_nr = dp_dir / f"validation{sfx_no_nr}.txt"
        te_no_nr = dp_dir / f"test{sfx_no_nr}.txt"

        c_file_nr = dp_dir / f"clusters{sfx_nr}.json"
        tr_nr = dp_dir / f"train{sfx_nr}.txt"
        va_nr = dp_dir / f"validation{sfx_nr}.txt"
        te_nr = dp_dir / f"test{sfx_nr}.txt"

        # Načtení dat bez NR
        if c_file_no_nr.exists():
            c_dict_0, p2c_0 = load_clusters_data(str(c_file_no_nr))
            sp_0, _ = load_splits_data(str(tr_no_nr), str(va_no_nr), str(te_no_nr))
            stats_0, _, _ = compute_cluster_split_statistics(c_dict_0, p2c_0, sp_0, {})
            tot_0 = stats_0['Total']
            p_counts_0 = f"{stats_0['Train']['proteins']}/{stats_0['Validation']['proteins']}/{stats_0['Test']['proteins']} ({tot_0['proteins']})"
            mean_str_0 = f"{tot_0['mean_size']:.2f} ± {tot_0['std_size']:.2f}"
            print(f"{tmscore:<10.1f} | {'Bez NR':<12s} | {p_counts_0:<25s} | {tot_0['clusters']:<8d} | {mean_str_0:<16s} | {tot_0['median_size']:<8.1f} | {'Baseline':<12s}")
        else:
            tot_0 = None

        # Načtení dat s NR
        if c_file_nr.exists():
            c_dict_nr, p2c_nr = load_clusters_data(str(c_file_nr))
            sp_nr, _ = load_splits_data(str(tr_nr), str(va_nr), str(te_nr))
            stats_nr, _, _ = compute_cluster_split_statistics(c_dict_nr, p2c_nr, sp_nr, {})
            tot_nr = stats_nr['Total']
            p_counts_nr = f"{stats_nr['Train']['proteins']}/{stats_nr['Validation']['proteins']}/{stats_nr['Test']['proteins']} ({tot_nr['proteins']})"
            mean_str_nr = f"{tot_nr['mean_size']:.2f} ± {tot_nr['std_size']:.2f}"

            if tot_0 and tot_0['proteins'] > 0:
                reduction_pct = (1.0 - (tot_nr['proteins'] / tot_0['proteins'])) * 100
                red_str = f"-{reduction_pct:.1f} %"
            else:
                red_str = "N/A"

            print(f"{tmscore:<10.1f} | {'NR 0.95':<12s} | {p_counts_nr:<25s} | {tot_nr['clusters']:<8d} | {mean_str_nr:<16s} | {tot_nr['median_size']:<8.1f} | {red_str:<12s}")
        else:
            tot_nr = None

        print("-" * 105)

        if tot_0 and tot_nr:
            comparison_rows.append({
                'tmscore': tmscore,
                'no_nr_proteins': tot_0['proteins'],
                'nr_proteins': tot_nr['proteins'],
                'protein_reduction_pct': round((1.0 - tot_nr['proteins'] / tot_0['proteins']) * 100, 2),
                'no_nr_clusters': tot_0['clusters'],
                'nr_clusters': tot_nr['clusters'],
                'no_nr_mean_cluster_size': round(tot_0['mean_size'], 2),
                'nr_mean_cluster_size': round(tot_nr['mean_size'], 2),
                'delta_mean_cluster_size': round(tot_nr['mean_size'] - tot_0['mean_size'], 2)
            })

    # Uložení do CSV / JSON
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_csv = out_path / "nr_cluster_comparison.csv"
        comp_df.to_csv(comp_csv, index=False)
        print(f"💾 Srovnávací tabulka klastrů uložena do: {comp_csv}")

        # Vykreslení grafu velikostí klastrů
        plot_nr_cluster_comparison(comp_df, out_path / "nr_vs_non_nr_clusters.png")

    return comparison_rows

def plot_nr_cluster_comparison(comp_df, output_path):
    """Vykreslí graf srovnání počtu proteinů a průměrné velikosti klastru."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    tmscores = comp_df['tmscore'].tolist()
    x = np.arange(len(tmscores))
    width = 0.35

    # 1. Počet proteinů
    ax1.bar(x - width/2, comp_df['no_nr_proteins'], width, label='Bez NR (Plný dataset)', color='#2E86AB', edgecolor='black', alpha=0.9)
    ax1.bar(x + width/2, comp_df['nr_proteins'], width, label='S NR 0.95 (Odstraněny homology)', color='#A23B72', edgecolor='black', alpha=0.9)
    ax1.set_title("Total Proteins Before and After NR 0.95", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("TM-Score Threshold", fontsize=11)
    ax1.set_ylabel("Number of Unique Structures", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"TM {t}" for t in tmscores])
    ax1.legend(loc='lower left', frameon=True)

    for i, r in comp_df.iterrows():
        ax1.annotate(f"-{r['protein_reduction_pct']:.0f}%",
                     xy=(i + width/2, r['nr_proteins']),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#A23B72')

    # 2. Průměrná velikost klastru
    ax2.bar(x - width/2, comp_df['no_nr_mean_cluster_size'], width, label='Bez NR (Mean Size)', color='#2E86AB', edgecolor='black', alpha=0.9)
    ax2.bar(x + width/2, comp_df['nr_mean_cluster_size'], width, label='S NR 0.95 (Mean Size)', color='#A23B72', edgecolor='black', alpha=0.9)
    ax2.set_title("Mean Proteins per Cluster (No-NR vs. NR 0.95)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("TM-Score Threshold", fontsize=11)
    ax2.set_ylabel("Mean Cluster Size (Proteins / Cluster)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"TM {t}" for t in tmscores])
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Graf srovnání klastrů uložen do: {output_path}")

def stage_3_run_benchmarks(target_splits, models, epochs=40, out_prefix="nr_benchmark_results", force=False):
    """
    Fáze 3: Spustí benchmark_all_models.py pro zadané modely a splity.
    """
    print("\n" + "=" * 80)
    print("FÁZE 3: SPUŠTĚNÍ MASTER BENCHMARKU (benchmark_all_models.py)")
    print(f"Vybrané splity ({len(target_splits)}): {target_splits}")
    print(f"Vybrané modely ({len(models)}): {models}")
    print(f"Počet epoch: {epochs} | Force přetrénování: {force}")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_all_models.py"),
        "--splits", *target_splits,
        "--models", *models,
        "--epochs", str(epochs),
        "--out-prefix", out_prefix
    ]
    if force:
        cmd.append("--force")

    print(f"Spouštím příkaz:\n{' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Benchmark byl úspěšně dokončen.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Chyba při běhu benchmark_all_models.py: {e}")
        return False
    return True

def stage_4_analyze_differences(out_prefix, tmscores, nr_threshold=0.95, models=None, out_dir=None):
    """
    Fáze 4: Analýza rozdílů ve výkonu modelů (Bez NR vs. S NR 0.95).
    """
    json_path = f"{out_prefix}.json"
    if not os.path.exists(json_path):
        print(f"⚠️ Soubor s výsledky {json_path} nenalezen. Nelze provést vyhodnocení.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    out_path = Path(out_dir) if out_dir else (PROJECT_ROOT / "data_statistics")
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 105)
    print(f"FÁZE 4: SROVNÁNÍ VÝKONU MODELŮ: BEZ NR vs. NR {nr_threshold} (HOMOLOGY BIAS ANALÝZA)")
    print("=" * 105)

    # Indexace výsledků podle (model, split)
    lookup = {}
    for entry in results_data:
        m = entry.get('model')
        sp = entry.get('split')
        if m and sp:
            lookup[(m, sp)] = entry

    eval_models = models if models and 'all' not in models else ALL_MODEL_KEYS
    delta_rows = []

    print(f"\n{'Model':<22s} | {'TM':<4s} | {'Test F1 (No-NR)':<16s} | {'Test F1 (NR)':<14s} | {'Δ F1':<10s} | {'Test Acc (No-NR)':<18s} | {'Test Acc (NR)':<14s} | {'Δ Acc':<10s}")
    print("-" * 120)

    for m in eval_models:
        for t in tmscores:
            sp_no_nr = f"mil_{t}"
            sp_nr = f"mil_{t}_nr{nr_threshold}"

            res_0 = lookup.get((m, sp_no_nr))
            res_nr = lookup.get((m, sp_nr))

            if not res_0 and not res_nr:
                continue

            f1_0 = res_0.get('test_macro_f1') if res_0 else None
            f1_nr = res_nr.get('test_macro_f1') if res_nr else None
            acc_0 = res_0.get('test_acc') if res_0 else None
            acc_nr = res_nr.get('test_acc') if res_nr else None

            # Výpočet delta (rozdílu)
            if f1_0 is not None and f1_nr is not None:
                delta_f1 = (f1_nr - f1_0) * 100
                f1_0_str = f"{f1_0 * 100:.2f} %"
                f1_nr_str = f"{f1_nr * 100:.2f} %"
                delta_f1_str = f"{delta_f1:+.2f} %"
            else:
                delta_f1 = None
                f1_0_str = f"{f1_0 * 100:.2f} %" if f1_0 is not None else "N/A"
                f1_nr_str = f"{f1_nr * 100:.2f} %" if f1_nr is not None else "N/A"
                delta_f1_str = "N/A"

            if acc_0 is not None and acc_nr is not None:
                delta_acc = (acc_nr - acc_0) * 100
                acc_0_str = f"{acc_0 * 100:.2f} %"
                acc_nr_str = f"{acc_nr * 100:.2f} %"
                delta_acc_str = f"{delta_acc:+.2f} %"
            else:
                delta_acc = None
                acc_0_str = f"{acc_0 * 100:.2f} %" if acc_0 is not None else "N/A"
                acc_nr_str = f"{acc_nr * 100:.2f} %" if acc_nr is not None else "N/A"
                delta_acc_str = "N/A"

            print(f"{m:<22s} | {t:<4.1f} | {f1_0_str:<16s} | {f1_nr_str:<14s} | {delta_f1_str:<10s} | {acc_0_str:<18s} | {acc_nr_str:<14s} | {delta_acc_str:<10s}")

            delta_rows.append({
                'model': m,
                'tmscore': t,
                'f1_no_nr': round(f1_0 * 100, 2) if f1_0 is not None else None,
                'f1_nr': round(f1_nr * 100, 2) if f1_nr is not None else None,
                'delta_f1': round(delta_f1, 2) if delta_f1 is not None else None,
                'acc_no_nr': round(acc_0 * 100, 2) if acc_0 is not None else None,
                'acc_nr': round(acc_nr * 100, 2) if acc_nr is not None else None,
                'delta_acc': round(delta_acc, 2) if delta_acc is not None else None
            })

    print("=" * 120 + "\n")

    if not delta_rows:
        print("⚠️ Žádná odpovídající data pro srovnání nebyla nalezena.")
        return

    # Exporty
    df_diff = pd.DataFrame(delta_rows)
    csv_out = out_path / "nr_vs_non_nr_diff.csv"
    json_out = out_path / "nr_vs_non_nr_diff.json"
    md_out = out_path / "nr_vs_non_nr_diff.md"

    df_diff.to_csv(csv_out, index=False)
    df_diff.to_json(json_out, orient="records", indent=4)

    # Markdown tabulka pro snadné zkopírování do prezentace / článku
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write("# Srovnání vlivu NR 0.95 filtrace na výkon AMICO modelů\n\n")
        f.write("| Model | TM-Score | Test F1 (No-NR) | Test F1 (NR 0.95) | Δ F1 | Test Acc (No-NR) | Test Acc (NR 0.95) | Δ Acc |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in delta_rows:
            f1_0_str = f"{r['f1_no_nr']:.2f} %" if r['f1_no_nr'] is not None else "N/A"
            f1_nr_str = f"{r['f1_nr']:.2f} %" if r['f1_nr'] is not None else "N/A"
            d_f1_str = f"**{r['delta_f1']:+.2f} %**" if r['delta_f1'] is not None else "N/A"
            acc_0_str = f"{r['acc_no_nr']:.2f} %" if r['acc_no_nr'] is not None else "N/A"
            acc_nr_str = f"{r['acc_nr']:.2f} %" if r['acc_nr'] is not None else "N/A"
            d_acc_str = f"{r['delta_acc']:+.2f} %" if r['delta_acc'] is not None else "N/A"
            f.write(f"| `{r['model']}` | {r['tmscore']} | {f1_0_str} | {f1_nr_str} | {d_f1_str} | {acc_0_str} | {acc_nr_str} | {d_acc_str} |\n")

    print(f"💾 Výsledné srovnání uloženo do:\n   - {csv_out}\n   - {json_out}\n   - {md_out}")

    # Vykreslení publikačního grafu
    plot_benchmark_nr_diff(df_diff, out_path / "nr_vs_non_nr_benchmark.png")

def plot_benchmark_nr_diff(df_diff, output_path):
    """Vykreslí přehledný 2-panelový graf srovnání Test Macro F1 (No-NR vs. NR 0.95)."""
    valid_df = df_diff.dropna(subset=['f1_no_nr', 'f1_nr'])
    if valid_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Průměrný výkon každého modelu napříč splity
    summary = valid_df.groupby('model').agg({
        'f1_no_nr': 'mean',
        'f1_nr': 'mean',
        'delta_f1': 'mean'
    }).reset_index()

    summary = summary.sort_values(by='f1_nr', ascending=False)
    models = summary['model'].tolist()
    x = np.arange(len(models))
    width = 0.38

    b1 = ax1.bar(x - width/2, summary['f1_no_nr'], width, label='Bez NR (Plný dataset)', 
                 color='#2E86AB', edgecolor='black', alpha=0.9)
    b2 = ax1.bar(x + width/2, summary['f1_nr'], width, label='S NR 0.95 (Odstraněny homology)', 
                 color='#A23B72', edgecolor='black', alpha=0.9)

    ax1.set_title("Average Test Macro F1 Across Models", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Model Architecture", fontsize=11, labelpad=8)
    ax1.set_ylabel("Average Test Macro F1 (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha='right', fontsize=9.5, fontweight='semibold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)

    for i, r in summary.iterrows():
        idx = models.index(r['model'])
        delta = r['delta_f1']
        color = '#2E86AB' if delta >= 0 else '#C73E1D'
        max_h = max(r['f1_nr'], r['f1_no_nr'])
        ax1.annotate(f"{delta:+.1f}%",
                     xy=(idx, max_h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)

    # Panel 2: Vývoj F1 skóre podle prahů TM-score (0.3 -> 0.9)
    top_models = [m for m in ['ligand_cross_mil', 'foldseek', 'sequence_mlp', 'cross_attention_mil'] if m in models]
    if not top_models:
        top_models = models[:4]

    palette = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    for idx_m, m in enumerate(top_models):
        m_df = valid_df[valid_df['model'] == m].sort_values(by='tmscore')
        t_vals = m_df['tmscore'].tolist()
        f1_0_vals = m_df['f1_no_nr'].tolist()
        f1_nr_vals = m_df['f1_nr'].tolist()
        col = palette[idx_m % len(palette)]

        ax2.plot(t_vals, f1_0_vals, marker='o', linewidth=2, linestyle='-', color=col, label=f"{m} (No-NR)")
        ax2.plot(t_vals, f1_nr_vals, marker='s', linewidth=2, linestyle='--', color=col, alpha=0.85, label=f"{m} (NR 0.95)")

    ax2.set_title("Test Macro F1 vs. TM-Score Clustering Threshold", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("TM-Score Threshold (Foldseek easy-cluster)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Test Macro F1 (%)", fontsize=11)
    ax2.set_xticks(sorted(valid_df['tmscore'].unique()))
    ax2.set_ylim(min(valid_df['f1_nr'].min() - 5, 20), 100)
    ax2.legend(loc='lower right', frameon=True, fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Publikační graf srovnání modelů uložen do: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Master skript: Generování NR 0.95 + 0.3/0.5/0.7/0.9 splitů, spuštění benchmarku a analýza rozdílů."
    )
    parser.add_argument(
        '--tmscores', nargs='+', type=float, default=DEFAULT_TMSCORES,
        help="Seznam prahů TM-score pro shlukování (default: 0.3 0.5 0.7 0.9)."
    )
    parser.add_argument(
        '--nr-threshold', type=float, default=DEFAULT_NR_THRESHOLD,
        help="Práh TM-score pro Non-Redundant předfiltraci (default: 0.95)."
    )
    parser.add_argument(
        '-d', '--structures-dir', default=None,
        help="Cesta ke složce se strukturami PDB."
    )
    parser.add_argument(
        '--models', nargs='+', default=['all'],
        help="Seznam modelů pro benchmark (nebo 'all')."
    )
    parser.add_argument(
        '--epochs', type=int, default=40,
        help="Počet trénovacích epoch pro benchmarkované modely (default: 40)."
    )
    parser.add_argument(
        '--out-prefix', default="nr_benchmark_results",
        help="Prefix pro výstupní soubory benchmarku (default: nr_benchmark_results)."
    )
    parser.add_argument(
        '--skip-clustering', action='store_true',
        help="Přeskočit generování klastrů (použít již vygenerované soubory)."
    )
    parser.add_argument(
        '--skip-benchmark', action='store_true',
        help="Přeskočit trénování a evaluaci modelů (pouze klastry a analýza)."
    )
    parser.add_argument(
        '--only-analysis', action='store_true',
        help="Přeskočit klastrování i benchmark a pouze vyhodnotit existující výsledky."
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Vynutit přegenerování klastrů i přetrénování již hotových modelů."
    )
    args = parser.parse_args()

    struct_dir = find_structures_directory(args.structures_dir)
    print("\n" + "=" * 80)
    print("       AMICO MASTER PIPELINE: NON-REDUNDANT (NR) EVALUATION       ")
    print("=" * 80)
    print(f"Detekovaná složka se strukturami: {struct_dir}")

    # 1. FÁZE: Generování splitů
    if not args.skip_clustering and not args.only_analysis:
        generated_splits = stage_1_generate_splits(
            tmscores=args.tmscores,
            nr_threshold=args.nr_threshold,
            structures_dir=struct_dir,
            output_dir=PROJECT_ROOT / "data_prep",
            force=args.force
        )
    else:
        generated_splits = []
        for t in args.tmscores:
            generated_splits.append(f"mil_{t}")
            generated_splits.append(f"mil_{t}_nr{args.nr_threshold}")

    # 2. FÁZE: Srovnání klastrů a redukce proteinů
    if not args.only_analysis:
        stage_2_compare_clusters(
            tmscores=args.tmscores,
            nr_threshold=args.nr_threshold,
            data_prep_dir=PROJECT_ROOT / "data_prep",
            out_dir=PROJECT_ROOT / "data_statistics"
        )

    # 3. FÁZE: Spuštění benchmarku všech modelů
    if not args.skip_benchmark and not args.only_analysis:
        target_models = ALL_MODEL_KEYS if 'all' in args.models else [m for m in args.models if m in ALL_MODEL_KEYS]
        success = stage_3_run_benchmarks(
            target_splits=generated_splits,
            models=target_models,
            epochs=args.epochs,
            out_prefix=args.out_prefix,
            force=args.force
        )

    # 4. FÁZE: Vyhodnocení rozdílů mezi No-NR a NR
    stage_4_analyze_differences(
        out_prefix=args.out_prefix,
        tmscores=args.tmscores,
        nr_threshold=args.nr_threshold,
        models=args.models,
        out_dir=PROJECT_ROOT / "data_statistics"
    )

if __name__ == '__main__':
    main()
