import json
import os
import glob
import csv
import re
import argparse
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import defaultdict

def is_aa(residue):
    return residue.get_id()[0] == ' '

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def compute_contact_map(ca_coords, threshold=8.0):
    if len(ca_coords) == 0:
        return []
    coords = np.array(ca_coords)
    dist_matrix = cdist(coords, coords)
    contact_map = (dist_matrix < threshold).astype(float)
    return contact_map.tolist()

def process_p2rank_outputs(structures_dir, output_json, min_prob=0.8):
    """
    Zpracuje výstupy z P2Ranku (_predictions.csv a _residues.csv) přímo z podsložek *_prank_output
    a vytvoří JSON dataset kapes pro MIL / EGNN modely.
    """
    supported_cofactors = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    # 1. Hledání všech _predictions.csv v podsložkách
    pred_files = glob.glob(os.path.join(structures_dir, '**', '*_predictions.csv'), recursive=True)
    
    # Fallback: pokud by náhodou existovaly fyzické *_pocket_*.pdb soubory
    pocket_pdb_files = glob.glob(os.path.join(structures_dir, '**', '*_pocket_*.pdb'), recursive=True)
    
    print(f"Nalezeno {len(pred_files)} P2Rank _predictions.csv souborů.")
    if pocket_pdb_files:
        print(f"Nalezeno také {len(pocket_pdb_files)} fyzických _pocket_*.pdb souborů.")
        
    if not pred_files and not pocket_pdb_files:
        print(f"Varování: V {structures_dir} nebyly nalezeny žádné výstupy z P2Ranku ani PDB kapsy.")
        return
        
    parser = PDBParser(QUIET=True)
    p2rank_dataset = []
    filtered_out_count = 0
    
    # Režim A: Zpracování ze standardních P2Rank CSV výstupů
    if pred_files:
        for pcsv in tqdm(pred_files, desc="Zpracovávám P2Rank výstupy"):
            csv_dir = os.path.dirname(pcsv)
            basename = os.path.basename(pcsv)
            
            # Zjištění ID proteinu
            # např. B7M2Z9_MERGED.pdb_predictions.csv -> B7M2Z9_MERGED
            prot_id = basename.replace('.pdb_predictions.csv', '').replace('_predictions.csv', '')
            
            # Zjištění kofaktoru (labelu) z cesty
            path_parts = Path(pcsv).parts
            cofactor_name = None
            for cof in supported_cofactors:
                if cof in path_parts:
                    cofactor_name = cof
                    break
                    
            if cofactor_name is None:
                continue
                
            label_idx = supported_cofactors.index(cofactor_name)
            
            # Hledání odpovídajícího full protein PDB souboru
            # Obvykle leží v nadřazené složce k *_prank_output
            parent_dir = os.path.dirname(csv_dir)
            pdb_candidates = [
                os.path.join(parent_dir, f"{prot_id}.pdb"),
                os.path.join(parent_dir, f"{prot_id}"),
                os.path.join(csv_dir, f"{prot_id}.pdb")
            ]
            
            pdb_file = None
            for cand in pdb_candidates:
                if os.path.exists(cand):
                    pdb_file = cand
                    break
                    
            if not pdb_file:
                # Rekurzivní hledání v rámci složky daného kofaktoru
                matches = [f for f in glob.glob(os.path.join(parent_dir, f"*{prot_id}*.pdb")) if '_pocket_' not in f]
                if matches:
                    pdb_file = matches[0]
                    
            if not pdb_file or not os.path.exists(pdb_file):
                # PDB proteinu nenalezeno
                continue
                
            # Načtení PDB struktury proteinu
            try:
                structure = parser.get_structure('protein', pdb_file)
                pdb_residues = {}
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if is_aa(residue):
                                chain_id = chain.get_id().strip()
                                resseq = str(residue.get_id()[1]).strip()
                                pdb_residues[(chain_id, resseq)] = residue
            except Exception as e:
                print(f"Chyba při čtení PDB {pdb_file}: {e}")
                continue
                
            # Načtení _predictions.csv pro získání pravděpodobností kapes
            pockets_info = {}
            try:
                with open(pcsv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, skipinitialspace=True)
                    for row in reader:
                        clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                        pname = clean_row.get('name', '')
                        prob = float(clean_row.get('probability', 0.0))
                        rank = clean_row.get('rank', '1')
                        
                        m = re.search(r'(\d+)', pname)
                        pnum = m.group(1) if m else str(rank)
                        pockets_info[pnum] = {'prob': prob, 'name': pname}
            except Exception as e:
                continue
                
            # Načtení _residues.csv pro mapování aminokyselin do jednotlivých kapes
            res_csv = pcsv.replace('_predictions.csv', '_residues.csv')
            if not os.path.exists(res_csv):
                res_csv = os.path.join(csv_dir, f"{prot_id}.pdb_residues.csv")
                
            if not os.path.exists(res_csv):
                continue
                
            pocket_residues = defaultdict(list)
            try:
                with open(res_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, skipinitialspace=True)
                    for row in reader:
                        clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                        pnum = clean_row.get('pocket', '0')
                        if pnum != '0' and pnum != '':
                            chain = clean_row.get('chain', '').strip()
                            res_label = clean_row.get('residue_label', '').strip()
                            pocket_residues[pnum].append((chain, res_label))
            except Exception:
                continue
                
            # Zpracování každé kapsy
            for pnum, res_list in pocket_residues.items():
                prob = pockets_info.get(pnum, {}).get('prob', 1.0)
                
                if prob < min_prob:
                    filtered_out_count += 1
                    continue
                    
                seq = []
                ca_coords = []
                
                for chain_id, res_label in res_list:
                    key = (chain_id, res_label)
                    if key in pdb_residues:
                        res = pdb_residues[key]
                        resname = res.get_resname()
                        if resname in three_to_one:
                            seq.append(three_to_one[resname])
                            if 'CA' in res:
                                ca_coords.append(res['CA'].get_coord().tolist())
                            else:
                                coords = [a.get_coord() for a in res.get_atoms()]
                                ca_coords.append(np.mean(coords, axis=0).tolist())
                                
                if len(seq) == 0:
                    continue
                    
                contact_map = compute_contact_map(ca_coords)
                pocket_id = f"{prot_id}_pocket_{pnum}"
                
                item = {
                    'pocket_id': pocket_id,
                    'protein_id': prot_id,
                    'label': label_idx,
                    'ligand_name': cofactor_name,
                    'binding_site_sequence': ''.join(seq),
                    'full_sequence': ''.join(seq),
                    'contact_map': contact_map,
                    'protein_coords': ca_coords,
                    'n_binding_site': len(seq),
                    'binding_site_indices': list(range(len(seq))),
                    'pdb_file': pdb_file,
                    'probability': prob
                }
                p2rank_dataset.append(item)
                
    print(f"\nÚspěšně zpracováno {len(p2rank_dataset)} kapes.")
    print(f"Vyfiltrováno {filtered_out_count} kapes s pravděpodobností < {min_prob}.")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(p2rank_dataset, f, indent=2)
        
    print(f"Uložen dataset do {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vytvoří JSON dataset kapes z výstupů P2Ranku")
    parser.add_argument("--structures-dir", "--dir", default="../structures", help="Cesta ke složce se strukturami")
    parser.add_argument("--output", default="p2rank_pockets_dataset.json", help="Výstupní JSON soubor")
    parser.add_argument("--min-prob", type=float, default=0.8, help="Minimální pravděpodobnost kapsy (default: 0.8)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.structures_dir):
        print(f"Složka {args.structures_dir} nebyla nalezena.")
        exit(1)
        
    process_p2rank_outputs(args.structures_dir, args.output, min_prob=args.min_prob)
