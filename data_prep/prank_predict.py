import os
import shutil
import subprocess
from pathlib import Path
import argparse

def sort_p2rank_outputs(pdb_filename, temp_out_dir, target_dir):
    """Vezme výsledky (CSV soubory) pro konkrétní PDB z dočasné složky a přesune je do cílové."""
    moved_anything = False
    
    # Najde všechny soubory začínající názvem PDB (např. 1fbl.pdb_predictions.csv)
    files_to_move = list(temp_out_dir.glob(f"{pdb_filename}*"))
    
    if files_to_move:
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files_to_move:
            if f.is_file():
                shutil.move(str(f), str(target_dir / f.name))
                moved_anything = True
                
    return moved_anything

def run_p2rank_batch(structures_dir, prank_exec="p2rank_2.5.1/prank", threads=6):
    structures_root = Path(structures_dir)
    temp_out_dir = Path("./temp_prank_out")
    ds_file = Path("current_batch.ds")

    if not structures_root.exists():
        raise FileNotFoundError(f"Složka neexistuje: {structures_root.resolve()}")

    # Zjistíme, co už je hotové a co chybí
    pdb_files = sorted(structures_root.rglob("*.pdb"))
    to_process = []

    for pdb_path in pdb_files:
        target_dir = Path(str(pdb_path.with_suffix("")) + "_prank_output")
        expected_csv = target_dir / f"{pdb_path.name}_predictions.csv"
        
        if target_dir.exists() and expected_csv.exists():
            continue  # Už je kompletně spočítáno
            
        to_process.append(pdb_path)

    print(f"Celkem PDB struktur: {len(pdb_files)}")
    print(f"Již hotovo: {len(pdb_files) - len(to_process)}")
    print(f"Zbývá spočítat: {len(to_process)}")

    if not to_process:
        print("Vše je kompletní!")
        return

    # Vytvoření dočasného .ds souboru (BEZ HLAVIČKY)
    print(f"Generuji {ds_file} pro {len(to_process)} struktur...")
    with open(ds_file, "w") as f:
        for pdb_path in to_process:
            f.write(f"{pdb_path.resolve()}\n")

    # Hromadné spuštění P2Ranku
    cmd = [
        prank_exec, "predict",
        "-c", "alphafold",
        "-threads", str(threads),
        "-visualizations", "0",
        "-o", str(temp_out_dir),
        str(ds_file)
    ]

    print("\nSpouštím P2Rank... (Interně běží paralelně, vizualizace jsou vypnuté)")
    try:
        subprocess.run(cmd, check=True)
        print("Výpočet P2Rank úspěšně dokončen.")
    except subprocess.CalledProcessError:
        print("\n[UPOZORNĚNÍ] P2Rank byl přerušen nebo spadl.")
        print("Python nyní zachrání a roztřídí soubory, které se stihly spočítat.\n")

    # Roztřídění dat zpět ke zdrojům
    print("Třídím výstupy do správných složek...")
    saved_count = 0
    
    for pdb_path in to_process:
        target_dir = Path(str(pdb_path.with_suffix("")) + "_prank_output")
        if sort_p2rank_outputs(pdb_path.name, temp_out_dir, target_dir):
            saved_count += 1
            
    print(f"Úspěšně zařazeno a uloženo {saved_count} struktur.")

    if ds_file.exists():
        ds_file.unlink()
        
    if temp_out_dir.exists():
        try:
            shutil.rmtree(temp_out_dir)
        except OSError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="../structures", help="Directory with PDB files")
    parser.add_argument("--prank", default="p2rank_2.5.1/prank", help="Path to P2Rank executable")
    parser.add_argument("--threads", type=int, default=6)
    
    args = parser.parse_args()
    run_p2rank_batch(args.dir, args.prank, args.threads)
