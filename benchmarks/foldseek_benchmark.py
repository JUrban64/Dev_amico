import os
import sys
import glob
import json
import argparse
import tempfile
import subprocess
import pandas as pd
import shutil
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset import load_split_ids

def get_pdb_paths(project_root):
    pdb_roots = [
        os.path.join(project_root, 'data_prep', 'structures'), 
        os.path.join(project_root, 'data_prep', 'Binding_Sites'),
        os.path.join(project_root, 'structures'),
        os.path.join(project_root, 'Binding_Sites'),
        os.path.join(project_root, '..', 'EquiPocket-MIL-', 'structures'),
        os.path.join(project_root, '..', 'EquiPocket-MIL-', 'Binding_Sites')
    ]
    pdb_files = {}
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                if '_pocket' not in p and 'prank_output' not in p:
                    base_id = os.path.basename(p).replace('.pdb', '')
                    if base_id not in pdb_files:
                        pdb_files[base_id] = p
    return pdb_files

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

def get_valid_dataset_pids(project_root):
    """Získá ID proteinů, které reálně existují v esm_dataset.pt (a esm_full_proteins.pt)."""
    dataset_candidates = [
        os.path.join(project_root, 'data_prep', 'esm_dataset.pt'),
        os.path.join(project_root, 'esm_dataset.pt')
    ]
    pockets_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if not pockets_path:
        return None
        
    try:
        import torch
        raw_pockets = torch.load(pockets_path, weights_only=False)
        pids = set()
        for item in raw_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            pids.add(pid)
            
        full_prot_path = os.path.join(os.path.dirname(pockets_path), 'esm_full_proteins.pt')
        if os.path.exists(full_prot_path):
            full_prots = torch.load(full_prot_path, weights_only=False)
            pids = pids.intersection(set(full_prots.keys()))
            
        return pids
    except Exception as e:
        print(f"Varování při načítání datasetu: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Foldseek 1-NN Benchmark")
    parser.add_argument("--split-suffix", "--suffix", default="mil_0.5", help="Přípona split souborů (např. mil_0.5, struct_pocket_0.5_0.5)")
    parser.add_argument("--use-esm-split", action="store_true", help="Použít ESM embedding clustering split (_esm_0.2)")
    parser.add_argument("--use-nr", action="store_true", help="Použít Non-Redundant (NR) variantu splitu")
    parser.add_argument("--all-pdbs", action="store_true", help="Vyhodnotit všechny PDB na disku (vypne automatické filtrování podle esm_dataset.pt)")
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'

    # 1. Načtení splitů pomocí robustní funkce
    train_ids_set, val_ids_set, test_ids_set = load_split_ids(project_root, split_suffix=args.split_suffix, use_nr=args.use_nr)
    train_ids = list(train_ids_set)
    test_ids = list(test_ids_set)
    
    if len(train_ids) == 0 or len(test_ids) == 0:
        print(f"Chyba: Soubory splitů pro suffix '{args.split_suffix}' nebyly nalezeny ve složce data_prep/.")
        print(f"Hledaný suffix: '{args.split_suffix}' (use_nr={args.use_nr})")
        return
        
    print(f"Načteno {len(train_ids)} train a {len(test_ids)} test proteinů (suffix: '{args.split_suffix}').")

    # Filtrování na proteiny, které jsou skutečně v datasetu modelu (jablka s jablky)
    if not args.all_pdbs:
        valid_pids = get_valid_dataset_pids(project_root)
        if valid_pids:
            orig_tr, orig_te = len(train_ids), len(test_ids)
            train_ids = [pid for pid in train_ids if match_id(pid, valid_pids)]
            test_ids = [pid for pid in test_ids if match_id(pid, valid_pids)]
            print(f"\n[FILTROVÁNO PODLE esm_dataset.pt - FÉROVÉ SROVNÁNÍ]:")
            print(f" - Train proteiny: {len(train_ids)} (z {orig_tr})")
            print(f" - Test proteiny:  {len(test_ids)} (z {orig_te})\n")
        else:
            print("Informace: esm_dataset.pt nenalezen, vyhodnocuji všechny PDB soubory.")

    # 3. Nalezení fyzických PDB souborů
    pdb_files = get_pdb_paths(project_root)
    
    # 2. Načtení labelů podle složky
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    name_to_label = {name: i for i, name in enumerate(target_names)}
    
    labels_by_pid = {}
    for pid, path in pdb_files.items():
        parts = os.path.normpath(path).split(os.sep)
        for part in reversed(parts):
            if part in name_to_label:
                labels_by_pid[pid] = name_to_label[part]
                break
                
    # Převažující třída v trénovací sadě (fallback, když Foldseek nenajde nic)
    train_labels = [labels_by_pid[pid] for pid in train_ids if pid in labels_by_pid]
    if len(train_labels) == 0:
        print("Chyba: Pro train proteiny nebyly nalezeny žádné labely.")
        return
        
    majority_train_label = Counter(train_labels).most_common(1)[0][0]
    print(f"Majority train label (použito pro fallback): {majority_train_label}")
    
    with tempfile.TemporaryDirectory(prefix="fs_bench_") as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train_pdbs")
        test_dir = os.path.join(tmp_dir, "test_pdbs")
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        
        print(f"Kopíruji PDB soubory do dočasné složky {tmp_dir} (může to trvat pár vteřin)...", flush=True)
        # Kopírování PDBček (použijeme symlink pro rychlost, pokud selže tak copy)
        copied_train = 0
        for pid in train_ids:
            if pid in pdb_files:
                try:
                    os.symlink(pdb_files[pid], os.path.join(train_dir, f"{pid}.pdb"))
                except OSError:
                    shutil.copy2(pdb_files[pid], os.path.join(train_dir, f"{pid}.pdb"))
                copied_train += 1
                
        copied_test = 0
        for pid in test_ids:
            if pid in pdb_files:
                try:
                    os.symlink(pdb_files[pid], os.path.join(test_dir, f"{pid}.pdb"))
                except OSError:
                    shutil.copy2(pdb_files[pid], os.path.join(test_dir, f"{pid}.pdb"))
                copied_test += 1
                
        print(f"Zkopírováno {copied_train} train a {copied_test} test PDB souborů.", flush=True)
        
        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        fs_tmp = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp)
        
        # Kontrola, jestli existuje foldseek
        if shutil.which("foldseek") is None:
            print("CHYBA: Příkaz 'foldseek' nebyl nalezen v systémové cestě (PATH)!", flush=True)
            return
        
        # 4. Spuštění Foldseeku (Test vs Train)
        cmd = [
            "foldseek", "easy-search",
            test_dir, train_dir, out_tsv, fs_tmp,
            "--format-output", "query,target,evalue,qtmscore,bits",
            "-e", "10.0",
            "--threads", "8"
        ]
        
        print("\nSpouštím Foldseek easy-search (Test vs Train)...", flush=True)
        try:
            subprocess.run(cmd, check=True)
            print("Foldseek úspěšně doběhl.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"CHYBA: Foldseek spadl s chybou: {e}", flush=True)
            return
        except Exception as e:
            print(f"CHYBA: Neočekávaná chyba při spouštění Foldseeku: {e}", flush=True)
            return
            
        # 5. Zpracování výsledků (1-Nearest Neighbor baseline)
        if not os.path.exists(out_tsv):
            print("Chyba: Foldseek nevygeneroval výstupní soubor.")
            return
            
        df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "evalue", "qtmscore", "bits"])
        df['query'] = df['query'].str.replace('.pdb', '', regex=False)
        df['target'] = df['target'].str.replace('.pdb', '', regex=False)
        
        # Foldseek řadí primárně podle evalue od nejlepšího (nejnižšího).
        # Přeradíme navíc podle qtmscore sestupně, jelikož chceme strukturálně nejpodobnější hit.
        df = df.sort_values(by=['query', 'qtmscore'], ascending=[True, False])
        
        # Ponecháme pouze první (Top-1) hit pro každý query protein
        top1_hits = df.drop_duplicates(subset=['query'], keep='first')
        
        y_true = []
        y_pred = []
        
        for pid in test_ids:
            if pid not in labels_by_pid:
                continue
                
            y_true.append(labels_by_pid[pid])
            
            hit = top1_hits[top1_hits['query'] == pid]
            if len(hit) > 0:
                target_pid = hit.iloc[0]['target']
                pred_label = labels_by_pid.get(target_pid, majority_train_label)
            else:
                # Pokud Foldseek nenašel žádný match ani s volným prahem, predikujeme fallback třídu
                pred_label = majority_train_label
                
            y_pred.append(pred_label)
            
        print("\n==========================================")
        print("      FOLDSEEK 1-NN BENCHMARK VÝSLEDKY     ")
        print("==========================================")
        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1_m:.4f}\n")
        
        print("Detailní report:")
        # Běžné target names v AMICO projektu
        target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD'] 
        
        try:
            print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        except ValueError:
            # Fallback pro jistotu, pokud jsou třídy jiné nebo jich je méně
            print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
