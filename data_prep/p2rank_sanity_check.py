import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import tqdm
from Bio.PDB import PDBParser
from pathlib import Path
import subprocess
import shutil

# Import extractor from the project to accurately find the GT ligand
from Binding_site_ex import BindingSiteExtractor

def is_hit(pocket_center, ligand_coords, threshold=4.0, metric='DCA'):
    center = np.array(pocket_center).reshape(1, 3)
    if metric == 'DCA':
        dists = cdist(center, ligand_coords)
        return np.min(dists) <= threshold
    elif metric == 'DCC':
        ligand_com = np.mean(ligand_coords, axis=0).reshape(1, 3)
        return cdist(center, ligand_com)[0][0] <= threshold
    return False

def sort_p2rank_outputs(pdb_filename, temp_out_dir, target_dir):
    moved_anything = False
    files_to_move = list(temp_out_dir.glob(f"{pdb_filename}*"))
    if files_to_move:
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files_to_move:
            if f.is_file():
                shutil.move(str(f), str(target_dir / f.name))
                moved_anything = True
    return moved_anything

def run_p2rank_batch(missing_pdbs, threads=6):
    temp_out_dir = Path("./temp_prank_out_sanity")
    ds_file = Path("sanity_batch.ds")
    
    print(f"Spouštím P2Rank pro {len(missing_pdbs)} struktur...")
    with open(ds_file, "w") as f:
        for pdb_path in missing_pdbs:
            f.write(f"{Path(pdb_path).resolve()}\n")
            
    cmd = [
        "p2rank_2.5.1/prank", "predict",
        "-threads", str(threads),
        "-visualizations", "0",
        "-o", str(temp_out_dir),
        str(ds_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[VAROVÁNÍ] P2Rank se přerušil, zkusím zpracovat to, co se stihlo.")
        
    print("Třídím P2Rank výstupy...")
    for pdb_path in missing_pdbs:
        pdb_p = Path(pdb_path)
        target_dir = pdb_p.with_suffix("")
        target_dir = Path(str(target_dir) + "_prank_output")
        sort_p2rank_outputs(pdb_p.name, temp_out_dir, target_dir)
        
    if ds_file.exists(): ds_file.unlink()
    if temp_out_dir.exists(): shutil.rmtree(temp_out_dir, ignore_errors=True)

def main():
    base_dir = Path("./Binding_Sites")
    out_txt = "p2rank_sanity_check_results.txt"
    DCA_THRESHOLD = 4.0
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return
        
    # Najít všechny .pdb a vyfiltrovat jen ty z 'positive' složek
    all_pdbs = list(base_dir.rglob("*.pdb"))
    pdb_files = [p for p in all_pdbs if "positi" in p.parent.name.lower()]
    
    print(f"Nalezeno {len(pdb_files)} GT struktur (positive samples).")
    if len(pdb_files) == 0:
        return
        
    # Krok 1: Kontrola P2Rank predikcí
    missing_predictions = []
    for pdb_path in pdb_files:
        basename = pdb_path.name
        target_dir = Path(str(pdb_path.with_suffix("")) + "_prank_output")
        expected_csv = target_dir / f"{basename}_predictions.csv"
        
        if not expected_csv.exists():
            missing_predictions.append(pdb_path)
            
    if missing_predictions:
        run_p2rank_batch(missing_predictions)
    
    # Krok 2: Analýza
    extractor = BindingSiteExtractor()
    parser = PDBParser(QUIET=True)
    
    results = {
        'total': 0,
        'has_predictions': 0,
        'ligand_found': 0,
        'top1_hits': 0,
        'top3_hits': 0,
        'top5_hits': 0,
        'any_hits': 0,
        'avg_dca_best': [],
        'per_ligand': {}
    }
    
    for pdb_path in tqdm.tqdm(pdb_files):
        results['total'] += 1
        
        basename = pdb_path.name
        target_dir = Path(str(pdb_path.with_suffix("")) + "_prank_output")
        prank_csv = target_dir / f"{basename}_predictions.csv"
        
        if not prank_csv.exists():
            continue
                
        results['has_predictions'] += 1
        
        # Získání správného ligandu (folder name je 2 úrovně nad)
        expected_ligand = pdb_path.parent.parent.name
        parsed_path, needs_cleanup = extractor._preprocess_pdb(str(pdb_path))
        
        try:
            structure = parser.get_structure('protein', parsed_path)
            actual_ligand = extractor._resolve_ligand_name(structure, expected_ligand)
            ligand_coords = extractor._get_ligand_coords(structure, actual_ligand)
        except Exception:
            ligand_coords = None
        finally:
            if needs_cleanup and os.path.exists(parsed_path):
                os.unlink(parsed_path)
            
        if ligand_coords is None or len(ligand_coords) == 0:
            continue
            
        results['ligand_found'] += 1
        
        if expected_ligand not in results['per_ligand']:
            results['per_ligand'][expected_ligand] = {
                'total': 0, 'top1': 0, 'top3': 0, 'top5': 0, 'any': 0
            }
        results['per_ligand'][expected_ligand]['total'] += 1
        
        try:
            df = pd.read_csv(prank_csv, skipinitialspace=True)
            df.columns = df.columns.str.strip()
        except:
            continue
            
        if len(df) == 0:
            continue
            
        hit_ranks = []
        best_dca = float('inf')
        
        for i, row in df.iterrows():
            pocket_center = [row['center_x'], row['center_y'], row['center_z']]
            dists = cdist(np.array(pocket_center).reshape(1,3), ligand_coords)
            min_dist = np.min(dists)
            if min_dist < best_dca:
                best_dca = min_dist
                
            if min_dist <= DCA_THRESHOLD:
                hit_ranks.append(i + 1)
                
        results['avg_dca_best'].append(best_dca)
        
        if len(hit_ranks) > 0:
            results['any_hits'] += 1
            results['per_ligand'][expected_ligand]['any'] += 1
            best_rank = hit_ranks[0]
            if best_rank <= 1:
                results['top1_hits'] += 1
                results['per_ligand'][expected_ligand]['top1'] += 1
            if best_rank <= 3:
                results['top3_hits'] += 1
                results['per_ligand'][expected_ligand]['top3'] += 1
            if best_rank <= 5:
                results['top5_hits'] += 1
                results['per_ligand'][expected_ligand]['top5'] += 1

    # Krok 3: Tvorba reportu
    summary = []
    summary.append("==================================================")
    summary.append("       P2RANK SANITY CHECK (Binding_Sites)")
    summary.append("==================================================")
    summary.append(f"Total GT structures checked: {results['total']}")
    summary.append(f"Structures with P2Rank predictions: {results['has_predictions']}")
    summary.append(f"Structures with resolvable GT ligand: {results['ligand_found']}")
    summary.append("--------------------------------------------------")
    
    if results['ligand_found'] > 0:
        n = results['ligand_found']
        summary.append(f"OVERALL Top-1 Success:   {results['top1_hits']:5d} / {n} ({(results['top1_hits']/n*100):.1f}%)")
        summary.append(f"OVERALL Top-3 Success:   {results['top3_hits']:5d} / {n} ({(results['top3_hits']/n*100):.1f}%)")
        summary.append(f"OVERALL Top-5 Success:   {results['top5_hits']:5d} / {n} ({(results['top5_hits']/n*100):.1f}%)")
        summary.append(f"OVERALL Any Hit Success: {results['any_hits']:5d} / {n} ({(results['any_hits']/n*100):.1f}%)")
        
        avg_dca = np.mean(results['avg_dca_best'])
        summary.append(f"Average minimum DCA:  {avg_dca:.2f} Å")
        
        summary.append("--------------------------------------------------")
        summary.append("PER LIGAND BREAKDOWN (Any Hit Success Rate):")
        for lig, stats in sorted(results['per_ligand'].items()):
            ltot = stats['total']
            lany = stats['any']
            l1 = stats['top1']
            l3 = stats['top3']
            summary.append(f" {lig:>10}: Any Hit: {lany:>4d}/{ltot:<4d} ({(lany/ltot*100):.1f}%) | Top-1: {(l1/ltot*100):.1f}% | Top-3: {(l3/ltot*100):.1f}%")
            
    else:
        summary.append("No ligands were successfully resolved.")
        
    summary.append("==================================================")
    summary.append("Metric used: DCA (Distance to Closest Atom) <= 4.0 A")
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open(out_txt, "w") as f:
        f.write(summary_text + "\n")
        
    print(f"\nVýsledky uloženy do: {out_txt}")

if __name__ == '__main__':
    main()
