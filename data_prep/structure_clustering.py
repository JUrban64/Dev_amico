import os
import json
import random
import shutil
import subprocess
import tempfile
import glob
import argparse
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))

def create_alias_pdb(src_pdb, dst_pdb):
    """Vytvoří fyzickou kopii (symlinky mohou dělat problémy na HPC/v kontejnerech)."""
    shutil.copy2(src_pdb, dst_pdb)

def find_foldseek_binary(explicit_path=None):
    """Najde cestu k binárce foldseek."""
    if explicit_path and os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
        return explicit_path
    
    # 1. Kontrola v PATH
    which_fs = shutil.which("foldseek")
    if which_fs:
        return which_fs
        
    # 2. Kontrola v běžných conda / miniforge prostředích
    candidates = [
        "/Users/jachymurban/miniforge3/envs/foldseek/bin/foldseek",
        os.path.expanduser("~/miniforge3/envs/foldseek/bin/foldseek"),
        os.path.expanduser("~/miniforge3/bin/foldseek"),
        os.path.expanduser("~/miniconda3/envs/foldseek/bin/foldseek"),
        os.path.expanduser("~/miniconda3/bin/foldseek"),
        "/opt/homebrew/bin/foldseek",
        "/usr/local/bin/foldseek"
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "foldseek"

def cluster_structures(target='both', test_limit=None, tmscore_threshold=0.5, nr_threshold=None, 
                       structures_dir=None, foldseek_bin=None):
    cwd = os.getcwd()
    pdb_roots = []
    if structures_dir:
        pdb_roots.append(os.path.abspath(structures_dir))
    
    if target == 'binding_sites':
        pdb_roots.extend([
            os.path.join(script_dir, 'Binding_Sites'),
            os.path.join(cwd, 'Binding_Sites')
        ])
    elif target == 'structures':
        pdb_roots.extend([
            os.path.join(script_dir, 'structures'),
            os.path.join(cwd, 'structures')
        ])
    else:
        pdb_roots.extend([
            os.path.join(script_dir, 'Binding_Sites'),
            os.path.join(script_dir, 'structures'),
            os.path.join(cwd, 'Binding_Sites'),
            os.path.join(cwd, 'structures')
        ])
    
    pdb_files = []
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                # Chceme clusterovat pouze full struktury, nikoliv výstřižky kapes
                if '_pocket' not in p and 'prank_output' not in p:
                    pdb_files.append(p)

    # Aplikace testovacího limitu
    if test_limit:
        pdb_files = pdb_files[:test_limit]
        print(f"--- Testovací režim zapnut: zpracovávám pouze {len(pdb_files)} struktur ---")

    with tempfile.TemporaryDirectory(prefix="fs_pdb_") as tmp_dir:
        pdb_data = {}
        tmp_pdb_dir = os.path.join(tmp_dir, "pdbs")
        os.makedirs(tmp_pdb_dir, exist_ok=True)
        
        for pdb_file in pdb_files:
            base_id = os.path.basename(pdb_file).replace(".pdb", "")
            
            if base_id in pdb_data:
                print(f"Warning: duplicate ID '{base_id}', skipping {pdb_file}")
                continue
                
            alias_pdb = os.path.join(tmp_pdb_dir, f"{base_id}.pdb")
            create_alias_pdb(pdb_file, alias_pdb)
            pdb_data[base_id] = alias_pdb

        print(f"Loaded {len(pdb_data)} unique PDB structures")
        if len(pdb_data) == 0:
            print("No PDBs found.")
            return None, None, None
        
        target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
        name_to_label = {name: str(i) for i, name in enumerate(target_names)}
        
        labels_by_pid = {}
        for pid, path in pdb_data.items():
            # Musíme dohledat originální cestu k PDB, pdb_data obsahuje cesty do temp diru,
            # takže použijeme původní pdb_files pole k nalezení originální cesty.
            pass
            
        # Vytvoříme si mapu z originálních PDB
        orig_labels_by_pid = {}
        for p in pdb_files:
            pid = os.path.basename(p).replace(".pdb", "")
            parts = os.path.normpath(p).split(os.sep)
            for part in reversed(parts):
                if part in name_to_label:
                    orig_labels_by_pid[pid] = name_to_label[part]
                    break
        
        labels_by_pid = orig_labels_by_pid
            
        fs_out_prefix = os.path.join(tmp_dir, "fs_out")
        fs_tmp_dir = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp_dir, exist_ok=True)
        
        fs_bin = find_foldseek_binary(foldseek_bin)
        print(f"Using Foldseek binary: {fs_bin}")

        # === 1. FÁZE: NON-REDUNDANT PRE-FILTRACE ===
        if nr_threshold is not None:
            print(f"\n=== Spouštím NR filtraci s prahem TM-score {nr_threshold} ===")
            nr_out_prefix = os.path.join(tmp_dir, "fs_nr_out")
            nr_tmp_dir = os.path.join(tmp_dir, "fs_nr_tmp")
            os.makedirs(nr_tmp_dir, exist_ok=True)
            
            nr_command = [
                fs_bin, "easy-cluster", 
                tmp_pdb_dir, nr_out_prefix, nr_tmp_dir,
                "--tmscore-threshold", str(nr_threshold),
                "--alignment-type", "2", 
                "-c", "0.8",
                "--threads", "8",
                "--cluster-mode", "1"
            ]
            
            try:
                subprocess.run(nr_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running Foldseek NR pre-filtering: {e}")
                return None, None, None
            except FileNotFoundError:
                print(f"Error: Foldseek executable '{fs_bin}' not found.")
                return None, None, None
                
            nr_cluster_tsv = f"{nr_out_prefix}_cluster.tsv"
            if not os.path.exists(nr_cluster_tsv):
                print(f"Error: NR Foldseek output {nr_cluster_tsv} not found.")
                return None, None, None
                
            nr_reps = set()
            with open(nr_cluster_tsv, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        rep = parts[0].replace('.pdb', '')
                        nr_reps.add(rep)
                        
            print(f"NR filtrace zredukovala dataset z {len(pdb_data)} na {len(nr_reps)} unikátních reprezentantů.")
            
            # Vytvoření nové složky pouze s NR reprezentanty pro další krok
            tmp_pdb_dir_nr = os.path.join(tmp_dir, "pdbs_nr")
            os.makedirs(tmp_pdb_dir_nr, exist_ok=True)
            
            for pid in nr_reps:
                if pid in pdb_data:
                    # symlink or copy to the new directory
                    try:
                        os.symlink(pdb_data[pid], os.path.join(tmp_pdb_dir_nr, f"{pid}.pdb"))
                    except OSError:
                        shutil.copy2(pdb_data[pid], os.path.join(tmp_pdb_dir_nr, f"{pid}.pdb"))
            
            # Nahrazení pracovní složky a slovníku s daty
            tmp_pdb_dir = tmp_pdb_dir_nr
            pdb_data = {k: v for k, v in pdb_data.items() if k in nr_reps}
        
        # === 2. FÁZE: HLAVNÍ SHLUKOVÁNÍ PRO TRAIN/TEST SPLIT ===
        print(f"\n=== Spouštím hlavní shlukování s prahem TM-score {tmscore_threshold} ===")
        
        command = [
            fs_bin, "easy-cluster", 
            tmp_pdb_dir, fs_out_prefix, fs_tmp_dir,
            "--tmscore-threshold", str(tmscore_threshold),
            "--alignment-type", "2", ## change to "2" for faster 3Di alphabet without TM-align 
            "-c", "0.8",
            "--threads", "8",
            "--cluster-mode", "1"
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Foldseek: {e}")
            return None, None, None
        except FileNotFoundError:
            print("Error: Foldseek executable not found. Please ensure it is installed and in your PATH.")
            return None, None, None
            
        cluster_tsv = f"{fs_out_prefix}_cluster.tsv"
        clusters = {}
        if not os.path.exists(cluster_tsv):
            print(f"Error: Foldseek output {cluster_tsv} not found.")
            return None, None, None
            
        with open(cluster_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep = parts[0].replace('.pdb', '')
                    member = parts[1].replace('.pdb', '')
                    if rep not in clusters:
                        clusters[rep] = []
                    clusters[rep].append(member)
                    
        print(f"Foldseek identified {len(clusters)} clusters.")

        # Mapování clusterů na jednotlivé proteiny
        sorted_pids = sorted(list(pdb_data.keys()))
        pid_to_cluster = {}
        for c_idx, (rep, members) in enumerate(clusters.items()):
            for m in members:
                pid_to_cluster[m] = c_idx
                
        cluster_labels = [pid_to_cluster.get(p, 0) for p in sorted_pids]
        protein_labels = np.array([int(labels_by_pid.get(p, -1)) for p in sorted_pids])
        
        print("\n" + "="*65)
        print("DIAGNOSTIKA CLUSTERŮ A TŘÍD")
        print("="*65)
        for label_idx, name in enumerate(target_names):
            pids_in_cls = [pid for pid in sorted_pids if labels_by_pid.get(pid) == str(label_idx)]
            clusters_in_cls = set([pid_to_cluster[p] for p in pids_in_cls if p in pid_to_cluster])
            cluster_sizes_in_cls = [len([p for p in pids_in_cls if pid_to_cluster.get(p) == c]) for c in clusters_in_cls]
            max_size = max(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            mean_size = np.mean(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            print(f"Třída {name:<12s}: {len(pids_in_cls):5d} proteinů v {len(clusters_in_cls):4d} clusterech (Max: {max_size:4d}, Průměr: {mean_size:.1f})")

        print("\n" + "-"*65)
        print("Rozdělování pomocí sklearn.model_selection.StratifiedGroupKFold (80/10/10)...")
        print("-" * 65)
        
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_assignments = np.zeros(len(sorted_pids), dtype=int)
        for fold_idx, (_, test_idx) in enumerate(sgkf.split(sorted_pids, protein_labels, groups=cluster_labels)):
            fold_assignments[test_idx] = fold_idx
            
        train_mask = fold_assignments < 8
        val_mask = fold_assignments == 8
        test_mask = fold_assignments == 9
        
        splits = {
            "train": [sorted_pids[i] for i in range(len(sorted_pids)) if train_mask[i]],
            "validation": [sorted_pids[i] for i in range(len(sorted_pids)) if val_mask[i]],
            "test": [sorted_pids[i] for i in range(len(sorted_pids)) if test_mask[i]]
        }
        
        print(f"\n{'Třída':<12s} | {'Train (80%)':<14s} | {'Val (10%)':<14s} | {'Test (10%)':<14s} | {'Celkem':<8s}")
        print("-" * 65)
        for label_idx, name in enumerate(target_names):
            lbl_str = str(label_idx)
            tr_c = sum(1 for p in splits["train"] if labels_by_pid.get(p) == lbl_str)
            va_c = sum(1 for p in splits["validation"] if labels_by_pid.get(p) == lbl_str)
            te_c = sum(1 for p in splits["test"] if labels_by_pid.get(p) == lbl_str)
            tot = tr_c + va_c + te_c
            tr_pct = (tr_c / tot * 100) if tot > 0 else 0
            va_pct = (va_c / tot * 100) if tot > 0 else 0
            te_pct = (te_c / tot * 100) if tot > 0 else 0
            print(f"{name:<12s} | {tr_c:5d} ({tr_pct:4.1f}%) | {va_c:5d} ({va_pct:4.1f}%) | {te_c:5d} ({te_pct:4.1f}%) | {tot:5d}")
            
        print("-" * 65)
        print(f"{'CELKEM':<12s} | {len(splits['train']):5d} ({(len(splits['train'])/len(sorted_pids)*100):4.1f}%) | {len(splits['validation']):5d} ({(len(splits['validation'])/len(sorted_pids)*100):4.1f}%) | {len(splits['test']):5d} ({(len(splits['test'])/len(sorted_pids)*100):4.1f}%) | {len(sorted_pids):5d}")
        print("=" * 65)
        
        return splits["train"], splits["validation"], splits["test"], clusters

def generate_clustering_splits(target='structures', tmscore_threshold=0.5, nr_threshold=None,
                               structures_dir=None, output_dir=None, foldseek_bin=None, test_limit=None):
    """
    Spustí shlukování a uloží train, validation, test a clusters.json soubory do výstupní složky.
    Vrací: (suffix, train, val, test, clusters)
    """
    train, val, test, clusters = cluster_structures(
        target=target, 
        test_limit=test_limit, 
        tmscore_threshold=tmscore_threshold,
        nr_threshold=nr_threshold,
        structures_dir=structures_dir,
        foldseek_bin=foldseek_bin
    )

    if train is None:
        return None, None, None, None, None

    target_suffix = ""
    if target == "binding_sites":
        target_suffix = "_e3"
    elif target == "structures":
        target_suffix = "_mil"
    elif target == "both":
        target_suffix = "_both"

    nr_suffix = f"_nr{nr_threshold}" if nr_threshold is not None else ""
    suffix = f"{target_suffix}_{tmscore_threshold}{nr_suffix}"

    save_dir = os.path.abspath(output_dir) if output_dir else script_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'train{suffix}.txt'), 'w') as f:
        for item in train:
            f.write(f"{item}\n")    

    with open(os.path.join(save_dir, f'validation{suffix}.txt'), 'w') as f:
        for item in val:
            f.write(f"{item}\n")
    
    with open(os.path.join(save_dir, f'test{suffix}.txt'), 'w') as f:
        for item in test:
            f.write(f"{item}\n")
            
    with open(os.path.join(save_dir, f'clusters{suffix}.json'), 'w') as f:
        json.dump(clusters, f, indent=4)
    
    print(f"Soubory uloženy do '{save_dir}': train{suffix}.txt, validation{suffix}.txt, test{suffix}.txt, clusters{suffix}.json")
    return suffix, train, val, test, clusters

if __name__ == "__main__":
    # Nastavení argparse
    parser = argparse.ArgumentParser(description="Cluster PDB structures using Foldseek.")
    parser.add_argument("--test", action="store_true", help="Omezí počet PDB souborů na 30 pro rychlé testování.")
    parser.add_argument("--target", choices=["binding_sites", "structures", "both"], default="structures", 
                        help="Co se má clustrovat: 'binding_sites' pro trénink E3, 'structures' pro MIL klasifikátor, nebo 'both'.")
    parser.add_argument("--tmscore-threshold", "--tmscore", type=float, default=0.5,
                        help="Práh TM-score pro Foldseek easy-cluster (default: 0.5).")
    parser.add_argument("--nr-threshold", type=float, default=None,
                        help="Práh TM-score pro prvotní Non-Redundant filtraci (např. 0.95). Pokud není zadáno, filtrace se neprovede.")
    parser.add_argument("-d", "--dir", "--structures-dir", dest="structures_dir", default=None,
                        help="Cesta ke složce se strukturami (pokud není zadána, hledá se v projektu).")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Výstupní složka pro uložení split souborů a klastrů (default: data_prep).")
    parser.add_argument("--foldseek-bin", default=None,
                        help="Explicitní cesta k binárce Foldseek.")
    args = parser.parse_args()

    # Určení limitu na základě argumentu
    limit = 30 if args.test else None

    # Spuštění a uložení
    generate_clustering_splits(
        target=args.target, 
        test_limit=limit, 
        tmscore_threshold=args.tmscore_threshold,
        nr_threshold=args.nr_threshold,
        structures_dir=args.structures_dir,
        output_dir=args.output_dir,
        foldseek_bin=args.foldseek_bin
    )