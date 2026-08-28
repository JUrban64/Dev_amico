import os
import sys
import glob
import argparse
import tempfile
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from dataset import load_split_ids

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def get_pdb_paths(project_root):
    """
    Najde všechny PDB soubory ve složkách projektu a zjistí jejich třídu ze složky.
    """
    pdb_roots = [
        os.path.join(project_root, 'data_prep', 'structures'), 
        os.path.join(project_root, 'structures'),
        
    ]
    pdb_files = {}
    pdb_classes = {}
    
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                if '_pocket' not in p and 'prank_output' not in p:
                    base_id = os.path.basename(p).replace('.pdb', '')
                    if base_id not in pdb_files:
                        pdb_files[base_id] = p
                        # Třída je název nadřazené složky (např. NAD, ATP...)
                        parent_folder = os.path.basename(os.path.dirname(p))
                        if parent_folder in TARGET_NAMES:
                            pdb_classes[base_id] = parent_folder
                        elif parent_folder.replace('_prank_output', '') in TARGET_NAMES:
                            pdb_classes[base_id] = parent_folder.replace('_prank_output', '')
    return pdb_files, pdb_classes

def run_foldseek_split_comparison(project_root, suffixes, use_nr=False, threads=8):
    pdb_files, pdb_classes = get_pdb_paths(project_root)
    print(f"Nalezeno {len(pdb_files)} PDB souborů.")
    
    records = []
    
    for suffix in suffixes:
        clean_suffix = suffix if suffix.startswith('mil_') or suffix.startswith('_') else f"mil_{suffix}"
        train_ids, val_ids, test_ids = load_split_ids(project_root, split_suffix=clean_suffix, use_nr=use_nr)
        
        if len(train_ids) == 0 or len(test_ids) == 0:
            print(f"Přeskakuji split '{suffix}': train nebo test soubor nenalezen.")
            continue
            
        print(f"\n--- Zpracovávám Foldseek TM-score pro split '{clean_suffix}' (use_nr={use_nr}) ---")
        
        # Filtrujeme dostupné PDB soubory
        train_pdbs = {pid: pdb_files[pid] for pid in train_ids if pid in pdb_files}
        test_pdbs = {pid: pdb_files[pid] for pid in test_ids if pid in pdb_files}
        
        print(f"Páruji PDB - Train: {len(train_pdbs)}/{len(train_ids)}, Test: {len(test_pdbs)}/{len(test_ids)}")
        
        if len(train_pdbs) == 0 or len(test_pdbs) == 0:
            continue
            
        with tempfile.TemporaryDirectory(prefix="fs_split_eval_") as tmp_dir:
            train_dir = os.path.join(tmp_dir, "train_pdbs")
            test_dir = os.path.join(tmp_dir, "test_pdbs")
            os.makedirs(train_dir)
            os.makedirs(test_dir)
            
            for pid, path in train_pdbs.items():
                shutil.copy2(path, os.path.join(train_dir, f"{pid}.pdb"))
            for pid, path in test_pdbs.items():
                shutil.copy2(path, os.path.join(test_dir, f"{pid}.pdb"))
                
            out_tsv = os.path.join(tmp_dir, "aln_results.tsv")
            fs_tmp = os.path.join(tmp_dir, "tmp")
            os.makedirs(fs_tmp, exist_ok=True)
            
            # Foldseek easy-search: Test (query) vs. Train (target)
            cmd = [
                "foldseek", "easy-search",
                test_dir, train_dir, out_tsv, fs_tmp,
                "--format-output", "query,target,fident,alnlen,qtmscore,alntmscore",
                "-e", "10.0",
                "--alignment-type", "1", # TM-align
                "--threads", str(threads)
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Chyba při spuštění Foldseek: {e}")
                return None
                
            if not os.path.exists(out_tsv) or os.path.getsize(out_tsv) == 0:
                print(f"Varování: Foldseek nevrátil žádné zarovnání pro {clean_suffix}.")
                continue
                
            df_aln = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "fident", "alnlen", "qtmscore", "alntmscore"])
            df_aln['query'] = df_aln['query'].str.replace('.pdb', '', regex=False)
            df_aln['target'] = df_aln['target'].str.replace('.pdb', '', regex=False)
            
            # Přidání tříd kofaktorů
            df_aln['q_class'] = df_aln['query'].map(pdb_classes)
            df_aln['t_class'] = df_aln['target'].map(pdb_classes)
            
            # Použijeme alntmscore nebo qtmscore
            tm_col = 'alntmscore' if 'alntmscore' in df_aln.columns else 'qtmscore'
            
            # Pro každý testovací protein spočítáme:
            # 1. Max TM-score v rámci stejné třídy
            # 2. Max TM-score k jiné třídě
            # 3. Max TM-score celkově
            
            for q_pid in test_pdbs.keys():
                q_hits = df_aln[df_aln['query'] == q_pid]
                q_class = pdb_classes.get(q_pid, 'Unknown')
                
                if len(q_hits) == 0:
                    max_same = 0.0
                    max_diff = 0.0
                    max_all = 0.0
                else:
                    same_hits = q_hits[q_hits['t_class'] == q_class]
                    diff_hits = q_hits[q_hits['t_class'] != q_class]
                    
                    max_same = same_hits[tm_col].max() if len(same_hits) > 0 else 0.0
                    max_diff = diff_hits[tm_col].max() if len(diff_hits) > 0 else 0.0
                    max_all = q_hits[tm_col].max()
                    
                records.append({
                    'Split': clean_suffix.replace('mil_', 'TM-práh '),
                    'Protein_ID': q_pid,
                    'Cofactor': q_class,
                    'Max_TM_Same_Class': max_same,
                    'Max_TM_Diff_Class': max_diff,
                    'Max_TM_Overall': max_all
                })
                
    return pd.DataFrame(records)

def plot_tmscore_results(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if df is None or len(df) == 0:
        print("Chyba: Žádná data pro vykreslení TM-score distribuce.")
        return
        
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Graf: Distribuce reálného Foldseek TM-score pro různé splity (0.0 až 1.0)
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("Set2", len(df['Split'].unique()))
    
    ax = sns.violinplot(
        data=df, 
        x='Split', 
        y='Max_TM_Overall', 
        palette=palette,
        inner="quartile",
        cut=0
    )
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.7, label='Práh 0.5 (homologní záhyb)')
    plt.title("Reálná Foldseek TM-score podobnost: Test vs. Train sada", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Práh klastrování splitu", fontsize=12, labelpad=10)
    plt.ylabel("Maximální Foldseek TM-score k nejbližšímu Train proteinu", fontsize=12, labelpad=10)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    
    out_file1 = os.path.join(out_dir, "foldseek_tmscore_split_violin.png")
    plt.savefig(out_file1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Graf distribuce TM-score uložen do: {out_file1}")
    
    # 2. Graf: Porovnání Same-Class vs. Diff-Class TM-score (Odhalení mezitřídního překryvu)
    df_melted = pd.melt(
        df,
        id_vars=['Split', 'Protein_ID', 'Cofactor'],
        value_vars=['Max_TM_Same_Class', 'Max_TM_Diff_Class'],
        var_name='Similarity_Type',
        value_name='TM_Score'
    )
    df_melted['Similarity_Type'] = df_melted['Similarity_Type'].map({
        'Max_TM_Same_Class': 'V rámci stejné třídy (Intra-class)',
        'Max_TM_Diff_Class': 'K cizím třídám (Inter-class / Shared Fold)'
    })
    
    g = sns.catplot(
        data=df_melted,
        x='Cofactor',
        y='TM_Score',
        hue='Similarity_Type',
        col='Split',
        kind='box',
        height=5,
        aspect=1.2,
        palette=['#4C72B0', '#C44E52'],
        fliersize=1.5
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Analýza strukturního překryvu: Stejná třída vs. Sdílené cizí rodiny (Rossmann fold)", fontsize=14, fontweight='bold')
    g.set_axis_labels("Kofaktor", "Foldseek TM-Score")
    
    out_file2 = os.path.join(out_dir, "foldseek_homology_leakage_breakdown.png")
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Detailní rozpad Same vs. Diff class uložen do: {out_file2}")

def main():
    parser = argparse.ArgumentParser(description="Výpočet reálného Foldseek TM-score mezi Test a Train sadami")
    parser.add_argument('--suffixes', nargs='+', default=['0.3', '0.5', '0.7', '0.9'], help='Splity k porovnání (např. 0.3 0.5 0.7 0.9)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant variantu splitů')
    parser.add_argument('--threads', type=int, default=8, help='Počet vláken pro Foldseek')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = run_foldseek_split_comparison(base_dir, args.suffixes, use_nr=args.use_nr, threads=args.threads)
    if df is not None:
        plot_tmscore_results(df, out_dir)

if __name__ == '__main__':
    main()
